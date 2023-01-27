import functions_framework
import zstd
from flask import jsonify, abort
from google.cloud import storage
import json


@functions_framework.http
def hello(request):
    if request.method == 'POST':
        name = request.form['name']
    else:
        name = request.args.get('name')

    if name is None:
        name = "unknown"

    response = {"greeting": "Hello {}!".format(name)}

    return jsonify(response)


@functions_framework.http
def listbucket(request):
    bucket, file_filter, uri_filter = get_params(request)
    if bucket is None:
        abort(404)

    blobs = get_blobs(bucket)
    response = []
    for blob in blobs:
        response.append(blob.name)

    return jsonify(response)


@functions_framework.http
def listfiles(request):
    bucket, file_filter, uri_filter = get_params(request)
    if bucket is None or file_filter is None:
        abort(404)

    blobs = get_blobs(bucket)
    response = []
    for blob in blobs:
        if blob.name.__contains__(file_filter):
            response.append(blob.name)

    return jsonify(response)


@functions_framework.http
def getlogs(request):
    bucket, file_filter, uri_filter = get_params(request)
    if bucket is None:
        abort(404)

    blobs = get_blobs(bucket)
    response = []
    for blob in blobs:
        if not blob.name.__contains__('.zst'):
            continue
        if file_filter is not None and not blob.name.__contains__(file_filter):
            continue
        data = blob.download_as_bytes()
        for jsonl in zstd.decompress(data).splitlines():
            json_data = json.loads(jsonl)
            if (json_data is not None and json_data['uri'] is not None
                    and json_data['uri'].__contains__(uri_filter)):
                response.append(json_data)

    return jsonify(response)


def get_params(request):
    if request.method == 'POST':
        bucket = request.form['bucket']
        file_filter = request.form['file_filter']
        uri_filter = request.form['uri_filter']
    else:
        bucket = request.args.get('bucket')
        file_filter = request.args.get('file_filter')
        uri_filter = request.args.get('uri_filter')
    return bucket, file_filter, uri_filter


def get_blobs(bucket):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket)
    return blobs
