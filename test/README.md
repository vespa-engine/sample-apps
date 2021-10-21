<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# README testing

Also see https://github.com/vespa-engine/documentation/tree/master/test

The URLs to test are specified in `_test_config.yml`.
Running

    $ ./test.py

without arguments will test each URL in that file in sequence.

To run this locally:

    $ python3 -m pip install -r requirements.txt
    $ ./test.py [URL | FILE]

If you want to run a test that is not in the above file, you can add the file
path or URL as an argument and that test will be run.


## Troubleshooting
Dump `vespa.log` - add to guide being tested:

    <pre data-test="exec">
    $ docker exec vespa bash -c 'cat /opt/vespa/logs/vespa/vespa.log'
    </pre>

`test.py` supports verbose output using `-v`, modify `screwdriver.yaml`:

    $SD_SOURCE_DIR/test/test.py -v -c $SD_SOURCE_DIR/test/_test_config.yml

Use the log download button top-right in the screwdriver.cd view
to make sure you see _all_ output.
