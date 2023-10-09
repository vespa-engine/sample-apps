# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json


def hello_handler(event, context):
    name = event.get('name', 'stranger')

    response = {"greeting": "Hello {}!".format(name)}

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
