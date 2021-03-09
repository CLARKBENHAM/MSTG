scrapping_options_prices.py
git add .
git commit -m "daily GME update"
git push
::for /L %i in (1,1,5) do timeout /T 60*60*24 & gme_everyday_script.bat