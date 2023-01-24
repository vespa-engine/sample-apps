import json


def hello_handler(event, context):
    name = event.get('name', 'stranger')

    response = {"greeting": "Hello {}!".format(name)}

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
