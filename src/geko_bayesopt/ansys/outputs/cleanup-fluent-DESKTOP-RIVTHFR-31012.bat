echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v261\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v261\fluent\ntbin\win64\tell.exe" DESKTOP-RIVTHFR 64315 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\ANSYSS~1\v261\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 33608) 
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 20256) 
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 31340) 
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 39124) 
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 31012) 
if /i "%LOCALHOST%"=="DESKTOP-RIVTHFR" (%KILL_CMD% 32404)
del "C:\Users\Galo\Documents\Matematicas\TUM (master)\Year 1\Case Study\clean_code\outputs\cleanup-fluent-DESKTOP-RIVTHFR-31012.bat"
