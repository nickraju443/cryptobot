=====================================================================
  CRYPTOBOT — SETUP (takes about 10 minutes, one time only)
=====================================================================

*********************************************************************
*  STEP 0 — EXTRACT THE ZIP FIRST. DO NOT SKIP THIS.                *
*                                                                   *
*  Right-click the zip file -> "Extract All..." -> Extract.        *
*  Then open the NEW folder it created.                             *
*  Running things from inside the zip window DOES NOT WORK.        *
*********************************************************************

BEFORE YOU START — you need 2 things installed:
  1. Google Chrome        -> https://www.google.com/chrome/
  2. Python 3.10 or newer -> https://www.python.org/downloads/
     *** Use that link — do NOT install Python from the Microsoft
         Store. During the install, CHECK THE BOX that says
         "Add python.exe to PATH". ***

IMPORTANT: If you have an OLD copy of this bot, DELETE that whole
folder first. The old copy is what causes the "insufficient funds"
and 401 errors. Use ONLY this new folder.


---------------------------------------------------------------------
STEP 1 — In the extracted folder, double-click START_EVERYTHING.bat
---------------------------------------------------------------------
- A Chrome window opens on the andX platform.
- A black console window installs the bot's packages (FIRST RUN ONLY,
  takes 5-15 minutes — let it finish, don't close it).
- The bot dashboard opens in your browser automatically
  (http://localhost:5002).
- A "CryptoBot" icon appears on your Desktop. From tomorrow on,
  that icon is all you ever click — it opens andX AND the dashboard.


---------------------------------------------------------------------
STEP 2 — Log into andX in the Chrome window
---------------------------------------------------------------------
- In the Chrome window that opened on platform.andx.one, log in with
  YOUR email and password.
- One time only — it remembers you next time.
- LEAVE THIS CHROME WINDOW OPEN. The bot places trades through it.
  If you close it, the bot cannot trade.


---------------------------------------------------------------------
STEP 3 — Create your andX API key
---------------------------------------------------------------------
- On the andX platform: account menu -> API Keys (or Settings -> API)
- Click "Create New API Key"
- Turn ON:  trading  AND  balance/wallet read permissions
- Set a passphrase (write it down — you need it in Step 4)
- Click Create.
- andX shows you:  API Key,  Secret,  Passphrase.
  *** The Secret is shown ONCE. Copy all 3 into a notepad NOW,
      before closing the page. ***


---------------------------------------------------------------------
STEP 4 — Connect the bot to YOUR account
---------------------------------------------------------------------
- Go to the bot dashboard:  http://localhost:5002
- Find the "andX API credentials" panel.
- Fill in:
     API key:       (from step 3)
     API secret:    (from step 3)
     Username:      your andX login email
     Passphrase:    (from step 3)
     Account name:  Main
- Click SAVE.
- Click TEST CONNECTION.
     GREEN + shows your balance  ->  you are DONE. Bot can trade.
     RED / error                 ->  see troubleshooting below.


---------------------------------------------------------------------
EVERYDAY USE
---------------------------------------------------------------------
- Start:  double-click the "CryptoBot" icon on your Desktop
          (Chrome opens already logged in, bot starts, dashboard opens)
- Stop:   close the black console window (or double-click STOP_BOT.bat)
- Rules:  keep the andX Chrome window OPEN while the bot runs.
          Your PC must stay ON for the bot to trade.


---------------------------------------------------------------------
TROUBLESHOOTING
---------------------------------------------------------------------
The console says "the bot's files are not next to this launcher":
  - You ran the .bat from inside the zip. Go back to STEP 0.

"Python is not installed" error:
  - Install Python from python.org (NOT the Microsoft Store) and
    CHECK "Add python.exe to PATH" during the install.
    Then run START_EVERYTHING.bat again.

"Could not create the Python environment":
  - Windows tricked you with a fake Store Python. Install the real
    one from https://www.python.org/downloads/ and try again.

"Test connection" fails / stays red:
  - Re-check every field for typos. The passphrase is the one you set
    when CREATING THE API KEY — it is NOT your andX login password.
  - Username = the email you log into andX with.
  - Account name must be exactly:  Main
  - Make sure the API key has trading + wallet permissions enabled.
  - Some accounts need the key confirmed by email — check your inbox.
  - Still failing? Delete the API key on andX and create a fresh one,
    then Save + Test again.

Bot says "insufficient funds" but I have money:
  - That means the bot cannot READ your balance = credentials problem.
    Fix "Test connection" first (above) and this goes away.

Chrome window closed by accident:
  - Just double-click the CryptoBot desktop icon again.

Dashboard didn't open:
  - Wait 30 seconds, then open  http://localhost:5002  manually.
=====================================================================
