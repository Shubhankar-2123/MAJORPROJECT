# Deployment Verification Checklist (Local)

Use this checklist to verify a clean setup using only the docs.

## 1) Environment
- [ ] Python 3.8–3.11 installed
- [ ] Create venv: `python -m venv .venv`
- [ ] Activate: `\.venv\Scripts\activate`
- [ ] Install deps: `pip install -r requirements.txt`

## 2) Run
- [ ] Start: `python flask_app/app.py`
- [ ] Browser: http://127.0.0.1:5000 → redirects to `/login`

## 3) Auth Flow
- [ ] Sign up a new user
- [ ] Login succeeds and redirects to `/app`
- [ ] Logout returns to `/login`

## 4) Predictions
- [ ] Upload image/video and get prediction
- [ ] Confirm prediction shows in `/dashboard/`

## 5) Feedback
- [ ] Use feedback button on a prediction
- [ ] Feedback appears in `/dashboard/`

## 6) Session
- [ ] Refresh `/app` while logged in (stays)
- [ ] Refresh `/app` while logged out (redirects to `/login`)
