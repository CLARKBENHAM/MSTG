@ECHO OFF
:run_script
    ::scrapping_options_prices.py > temp_file.txt
    
   setlocal
    %@Try%
        scrapping_options_prices.py
    %@EndTry%
    :@Catch ::runs regardless
        if %errorlevel%==1 (
         echo "Will Retry"
         timeout /T  3600
         goto run_script
         )
    :@EndCatch

git add .
git commit -m "daily GME update"
git push
for /L %i in (1,1,5) do timeout /T  86400 & gme_everyday_script.bat