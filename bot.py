import download
import os
import sys
import json
from datetime import datetime
from model_predict import get_response
import requests
from flask import Flask, request
import key

VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN', key.VERIFY_TOKEN)
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN', key.ACCESS_TOKEN)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():
    
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args.get('hub.challenge'), 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events

    data = request.get_json()
    #log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's
                    message_text = messaging_event["message"]["text"]  # the message's text
                    print("text recieved:", message_text)
                    response = get_response(message_text,0.7)
                    print("response generated:",response) 

                    send_message(sender_id, response)

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):
    print("sending messege to client")

    #log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "recipient": {
            "id": recipient_id
        },
        "message":{
            "text": message_text
        }
    }
    url = "https://graph.facebook.com/v2.6/me/messages?access_token="+key.ACCESS_TOKEN
    print("url:", url)
    #r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    #r = requests.post(url, headers=headers,  json=data)
    r = requests.post(url,json=data)
    if r.status_code != 200:
        print ("status code:",r.status_code)
        print("headers:",r.headers)
        print ("response text:", r.text)
        #log(r.status_code)
        #log(r.text)


'''
def log(msg, *args, **kwargs):  # simple wrapper for logging to stdout on heroku
    try:
        if type(msg) is dict:
            msg = json.dumps(msg)
        else:
            msg = unicode(msg).format(*args, **kwargs)
        print (u"{}: {}".format(datetime.now(), msg))
    except UnicodeEncodeError:
        pass  # squash logging errors in case of non-ascii text
    sys.stdout.flush()
'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
