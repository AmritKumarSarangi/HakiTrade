@echo off
echo 🚀 Starting HakiTrade Pre-Deployment Checks...

:: 1. Update Backend Requirements
echo 📂 Updating requirements.txt...
call .\venv\Scripts\pip freeze > requirements.txt
echo ✅ requirements.txt updated.

:: 2. Test Frontend Build
echo 🏗️ Testing Frontend Build...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo ❌ Frontend build failed! Check for errors in App.jsx or App.css.
    cd ..
    exit /b %errorlevel%
)
cd ..

echo ------------------------------------------------
echo ✨ HakiTrade is ready for deployment!
echo 1. Install Git from https://git-scm.com/
echo 2. Push to GitHub: git add . && git commit -m "Release v1.0" && git push
echo ------------------------------------------------
pause
