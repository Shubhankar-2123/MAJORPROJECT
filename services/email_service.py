"""
Email Service for SignAI
Handles password reset emails and notifications
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmailService:
    """Handles email sending for the application."""
    
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
    ):
        """
        Initialize email service.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address (from env or param)
            sender_password: Sender password/app password (from env or param)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL", "")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD", "")
        self.is_configured = bool(self.sender_email and self.sender_password)
    
    def is_available(self) -> bool:
        """Check if email service is properly configured."""
        return self.is_configured
    
    def send_password_reset_email(
        self,
        recipient_email: str,
        username: str,
        reset_link: str,
        verification_code: Optional[str] = None,
        expires_in_minutes: int = 15,
    ) -> Tuple[bool, str]:
        """
        Send password reset email.
        
        Args:
            recipient_email: Email address of recipient
            username: Username of the user
            reset_link: Full password reset link
            verification_code: One-time verification code for reset confirmation
            expires_in_minutes: Link expiration time
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_available():
            return False, "Email service not configured"
        
        try:
            subject = "SignAI - Password Reset Request"
            code_section_html = ""
            code_section_text = ""
            if verification_code:
                code_section_html = f"""
                            <div style=\"margin-top: 20px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px;\">
                                <p style=\"margin: 0 0 8px 0; font-size: 14px; color: #334155;\"><strong>Verification Code</strong></p>
                                <p style=\"margin: 0; font-size: 28px; letter-spacing: 4px; font-weight: bold; color: #0f172a;\">{verification_code}</p>
                                <p style=\"margin: 8px 0 0 0; font-size: 12px; color: #64748b;\">Enter this code on the reset page to verify your identity.</p>
                            </div>
                """
                code_section_text = f"""
            Verification code: {verification_code}
            Enter this code on the reset page to verify your identity.
                """
            
            # HTML email template
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                        <div style="background-color: #1e293b; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center;">
                            <h1 style="margin: 0; font-size: 28px;">🤟 SignAI</h1>
                            <p style="margin: 5px 0 0 0; font-size: 14px;">Sign Language Translation System</p>
                        </div>
                        
                        <div style="padding: 30px;">
                            <h2 style="color: #1e293b; margin-top: 0;">Password Reset Request</h2>
                            
                            <p>Hello <strong>{username}</strong>,</p>
                            
                            <p>We received a request to reset the password for your SignAI account. If you did not make this request, you can safely ignore this email.</p>
                            
                            <p style="margin-top: 30px;">
                                <a href="{reset_link}" style="display: inline-block; background-color: #0f766e; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                                    Reset Password
                                </a>
                            </p>
                            
                            <p style="margin-top: 20px; font-size: 12px; color: #666;">
                                Or copy this link: <br/>
                                <code style="background-color: #f0f0f0; padding: 10px; display: block; word-break: break-all;">{reset_link}</code>
                            </p>
                            
                            <p style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
                                <strong>⏱️ Security Note:</strong> This reset link will expire in {expires_in_minutes} minutes.
                            </p>
                            {code_section_html}
                            
                            <p style="color: #666; font-size: 12px;">
                                If you didn't request this, your account remains secure and no changes have been made.
                            </p>
                        </div>
                        
                        <div style="background-color: #f8f9fa; padding: 20px; border-top: 1px solid #ddd; text-align: center; font-size: 12px; color: #666; border-radius: 0 0 8px 8px;">
                            <p style="margin: 0;">© 2024 SignAI. All rights reserved.</p>
                            <p style="margin: 5px 0 0 0;">This is an automated message, please do not reply to this email.</p>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            # Plain text fallback
            text_content = f"""
            SignAI - Password Reset Request
            
            Hello {username},
            
            We received a request to reset the password for your SignAI account.
            
            Click the link below to reset your password (expires in {expires_in_minutes} minutes):
            {reset_link}
            {code_section_text}
            
            If you did not request this, you can safely ignore this email.
            
            © 2024 SignAI. All rights reserved.
            """
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.sender_email
            msg["To"] = recipient_email
            msg["Subject"] = subject
            
            # Attach both plain text and HTML
            msg.attach(MIMEText(text_content.strip(), "plain"))
            msg.attach(MIMEText(html_content.strip(), "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Password reset email sent to {recipient_email}")
            return True, "Password reset email sent successfully"
        
        except Exception as e:
            logger.error(f"Error sending password reset email: {str(e)}")
            return False, f"Error sending email: {str(e)}"
    
    def send_welcome_email(
        self,
        recipient_email: str,
        username: str,
    ) -> Tuple[bool, str]:
        """
        Send welcome email to new user.
        
        Args:
            recipient_email: Email address of recipient
            username: Username of the new user
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.is_available():
            return False, "Email service not configured"
        
        try:
            subject = "Welcome to SignAI!"
            
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                        <div style="background-color: #1e293b; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center;">
                            <h1 style="margin: 0; font-size: 28px;">🤟 SignAI</h1>
                            <p style="margin: 5px 0 0 0; font-size: 14px;">Sign Language Translation System</p>
                        </div>
                        
                        <div style="padding: 30px;">
                            <h2 style="color: #1e293b; margin-top: 0;">Welcome, {username}!</h2>
                            
                            <p>Welcome to SignAI! We're excited to have you on board.</p>
                            
                            <h3 style="color: #0f766e;">Getting Started</h3>
                            <ul style="margin: 20px 0; padding-left: 20px;">
                                <li><strong>Text → Sign Translation:</strong> Convert text to sign language</li>
                                <li><strong>Sign → Text Recognition:</strong> Recognize sign language from video</li>
                                <li><strong>Dictionary:</strong> Browse our sign language dictionary</li>
                                <li><strong>Multimodal Chat:</strong> Interactive chat with signs</li>
                                <li><strong>Custom Signs:</strong> Create and save custom signs</li>
                            </ul>
                            
                            <p style="background-color: #f0f9ff; padding: 15px; border-left: 4px solid #0f766e; border-radius: 4px;">
                                📖 Check out our <strong>Preview</strong> page to see system statistics and example demos.
                            </p>
                        </div>
                        
                        <div style="background-color: #f8f9fa; padding: 20px; border-top: 1px solid #ddd; text-align: center; font-size: 12px; color: #666; border-radius: 0 0 8px 8px;">
                            <p style="margin: 0;">© 2024 SignAI. All rights reserved.</p>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            text_content = f"""
            Welcome to SignAI!
            
            Hello {username},
            
            Welcome to SignAI! We're excited to have you on board.
            
            Getting Started:
            - Text → Sign Translation: Convert text to sign language
            - Sign → Text Recognition: Recognize sign language from video
            - Dictionary: Browse our sign language dictionary
            - Multimodal Chat: Interactive chat with signs
            - Custom Signs: Create and save custom signs
            
            Check out the Preview page to see system statistics and example demos.
            
            © 2024 SignAI. All rights reserved.
            """
            
            msg = MIMEMultipart("alternative")
            msg["From"] = self.sender_email
            msg["To"] = recipient_email
            msg["Subject"] = subject
            
            msg.attach(MIMEText(text_content.strip(), "plain"))
            msg.attach(MIMEText(html_content.strip(), "html"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Welcome email sent to {recipient_email}")
            return True, "Welcome email sent successfully"
        
        except Exception as e:
            logger.error(f"Error sending welcome email: {str(e)}")
            return False, f"Error sending email: {str(e)}"


# Global email service instance
email_service = EmailService()
