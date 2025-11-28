
<h1 align="center" style="margin: 0 auto 0 auto;"> 
   <img width="32" src="https://lookatwallstreet.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F0472a71b-02f2-43f2-b650-2ae94ae1fb5b%2Fc0e93390-aca9-4f7a-8b36-8a66ec8d925f%2F%25E5%25BE%25AE%25E4%25BF%25A1%25E6%2588%25AA%25E5%259B%25BE_20240930173619.png?table=block&id=1296853c-146c-8096-bb90-d38181edfea5&spaceId=0472a71b-02f2-43f2-b650-2ae94ae1fb5b&width=600&userId=&cache=v2" alt="logo" >  
   TradingBot Based on MooMoo/Futu
</h1>
<h4 align="center" style="margin: 0 auto 0 auto;">
   
 Build your own strategy following the example, let the bot trade for you! 

 Get market data, analyze data, make decision, place market or limit order, check account info, position status.

 All in One!
   
</h4>

- Make your code employed
- Make your bot employed 
- Make yourself free


## 1. Step by Step Setup:
1. Install Python: https://www.python.org/downloads/
   
3. Install MooMoo/Futu OpenD: https://www.moomoo.com/download/OpenAPI
   
5. Install requirements: run in CMD under root directory:
```
pip install -r requirements.txt
```

4. Go to the `env/_secret.py` and fill in your trading password.
   
5. Go to the `strategy/Your_Strategy.py`, and replace the example strategy in `strategy_decision(self)` with your own strategy.
   
6. Login to MooMoo/Futu OpenD, setup the port number to `11112` or `11111`, which should be the same as `MOOMOOOPEND_PORT` in `TradingBOT.py`.

<img src="https://github.com/user-attachments/assets/954d6846-32ed-457a-8508-b9b20b2e33d7" alt="image" width="800">

<br>

7. Check all the settings in `TradingBOT.py`, following the instructions in the project setup section.

8. Go to the project root directory and run in CMD:
```
python TradingBOT.py
```

<br>
