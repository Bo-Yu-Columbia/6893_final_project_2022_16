import json
import boto3
import time


# This lambda function is used to receive message from Lex and send back to front end
# It works as an intermediate layer between client (browser) and lex chatbot

# Main handler of lambda function
def lambda_handler(event, context):
    # text: message from front end
    text = get_request(event)
    session_id = event["sessionId"]

    # error handler: if no text inside request
    if text is None:
        return response_404("Error: Failed to get text from request.")

    # get chatbot response
    messages, session_state, request_attributes = get_chatbot_response(text, session_id)

    # error handler: if no response from lex
    if messages is None:
        return response_404("Error: Failed to connect with Lex.")
    else:
        # TODO: current only accept text as response message, need support of response card
        texts = []
        for message in messages:
            if message["contentType"] == "PlainText":
                texts.append(message["content"])
            elif message["contentType"] == "ImageResponseCard":
                texts.append(message["imageResponseCard"]["subtitle"])
            else:
                return response_404("Error: Unrecognizable content type.")

        return response_200(texts, session_id, session_state, request_attributes)


# To get text from request (Only accept text as input)
def get_request(event):
    # corner cases handle
    if "messages" not in event:
        return None
    messages = event["messages"]

    if not isinstance(messages, list) or len(messages) == 0:
        return None
    message = messages[0]

    if "unstructured" not in message or "text" not in message["unstructured"]:
        return None

    return message["unstructured"]["text"]


# successful response (200)
def response_200(texts, session_id, session_state, request_attributes):
    res = {
        "status code": 200,
        "messages": [
            {
                "type": "unstructured",
                "unstructured": {
                    "uid": session_id,
                    "text": text,
                    "time": time.time()
                }
            } for text in texts],
        "sessionState": session_state,
        "requestAttributes": request_attributes,
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }
    return res


# Unsuccessful response (404)
def response_404(message):
    res = {
        "status code": 404,
        "message": message
    }
    return res


# call lex api to get response
def get_chatbot_response(text, session_id, session_state=None):
    client = boto3.client('lexv2-runtime')

    params = {
        "botId": "POZUK7GBOG",
        "botAliasId": "ERWBY10NNS",
        "localeId": "en_US",
        "sessionId": session_id,
        "text": text
    }

    if session_state is not None:
        params["sessionState"] = session_state
    # pass message to Lex
    lex_response = client.recognize_text(**params)

    if not isinstance(lex_response, dict):
        return None

    if 'messages' not in lex_response:
        return None

    # get response from Lex
    return lex_response['messages'], lex_response.get("sessionState"), lex_response.get("requestAttributes")
