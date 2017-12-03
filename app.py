import os
import sys
import json
from datetime import datetime
from model_predict import get_response
import requests
from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events

    data = request.get_json()
    log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    response = get_response(message_text,0.7)

                    send_message(sender_id, response)

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": os.environ["PAGE_ACCESS_TOKEN"]
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


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

def download(path,store):
    from six.moves import urllib

    fname = store+path.split('/')[-1]

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0), end='\r')

    filepath, _ = urllib.request.urlretrieve(path, filename=fname, reporthook=progress)
    return filepath


if __name__ == '__main__':
    
    print ("checking requirement files....")

if os.path.exists('sequence2sequence/checkpoints/checkpoint'):
    print("checkpoints files present")
else:
    print('downloading checkpoint files')
    check_list = ['chatbot-1165000.data-00000-of-00001','chatbot-1165000.index','chatbot-1165000.meta','checkpoint']
    for item in check_list:
        url = "https://s3-us-west-1.amazonaws.com/cmpe297-checkpoint/"+item
        download(url,'sequence2sequence/checkpoints/')


if os.path.exists('sequence2sequence/processed/test.dec'):
    print("processed files present")
else:
    print('downloading processed files')
    check_list = ['test.dec','test.enc','test_ids.dec','test_ids.enc','train.dec','train.enc','train_ids.dec',
                  'train_ids.enc','vocab.dec','vocab.enc']
    for item in check_list:
        url = "https://s3-us-west-1.amazonaws.com/cmpe297-checkpoint/"+item
        download(url,'sequence2sequence/processed/')

    app.run(debug=True)
