#! /usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import io
import os
import sys
import getopt
import json
import time
import yaml
import urllib.request
import tempfile
import re

from bs4 import BeautifulSoup

from pseudo_terminal import PseudoTerminal

################################################################################
# Execution
################################################################################

verbose = False
workdir = "."
project_root = os.getcwd()
work_dir = os.path.join(project_root, "_work")
liquid_transforms = {}


def print_cmd_header(cmd, extra="", print_header=True):
    if not print_header:
        return
    print("")
    print("*" * 80)
    print("* {0}".format(cmd))
    if len(extra) > 0:
        print("* ({0})".format(extra))
    print("*" * 80)


def exec_wait(cmd, pty):
    command = cmd["$"]
    expect = cmd["wait-for"]
    max_wait = 300 if not ("timeout" in cmd) else int(cmd["timeout"])
    try_interval = 5  # todo: max this configurable too
    print_cmd_header(command, "Waiting for '{0}'".format(expect))

    waited = 0
    output = ""
    while waited < max_wait:
        exit_code, output = pty.run(command, verbose)
        if output.find(expect) >= 0:
            return
        else:
            time.sleep(try_interval)
            waited += try_interval
            print("Waited for {0}/{1} seconds...".format(waited, max_wait))

    if waited >= max_wait:
        if not verbose:
            print(output)
        raise RuntimeError("Expected output '{0}' not found in command '{1}'. Waited for {2} seconds.".format(expect, command, max_wait))


def exec_assert(cmd, pty):
    command = cmd["$"]
    expect = cmd["contains"]
    print_cmd_header(command, "Expecting '{0}'".format(expect))

    _, output = pty.run(command, verbose)
    if output.find(expect) == -1:
        if not verbose:
            print(output)
        raise RuntimeError("Expected output '{0}' not found in command '{1}'".format(expect, command))


def exec_file(cmd, pty):
    path = cmd["path"]
    print_cmd_header(path)
    path_array = []
    for dir in path.split(os.path.sep)[:-1]:
        path_array.append(dir)
        if not os.path.isdir(os.path.sep.join(path_array)):
            os.makedirs(os.path.sep.join(path_array))
    with open(str(path), "w") as f:
        f.write(str(cmd["content"]))

    print("Wrote " + str(len(cmd["content"])) + " chars to " + path)


def exec_expect(cmd, pty):
    command = cmd["$"]
    expect = cmd["expect"]
    timeout = cmd["timeout"]
    print_cmd_header(command, "Expecting '{0}'".format(expect))

    exit_code, output = pty.run_expect(command, expect, timeout, verbose)
    if exit_code != 0:
        if not verbose:
            print(output)
        raise RuntimeError("Command '{0}' returned code {1}".format(command, exit_code))


def exec_default(cmd, pty):
    command = cmd["$"]
    print_cmd_header(command)

    exit_code, output = pty.run(command, verbose)
    if exit_code != 0:
        if not verbose:
            print(output)
        raise RuntimeError("Command '{0}' returned code {1}".format(command, exit_code))


def exec_step(cmd, pty):
    globals()["exec_" + cmd["type"]](cmd, pty)


def exec_script(script):
    tmpdir = tempfile.mkdtemp(dir=work_dir)
    os.chdir(tmpdir)

    failed = False

    with PseudoTerminal(timeout=2*60*60) as pty:
        try:
            for cmd in script["before"]:
                exec_step(cmd, pty)
            for cmd in script["steps"]:
                exec_step(cmd, pty)
        except Exception as e:
            sys.stderr.write("ERROR: {0}\n".format(e))
            failed = True
        finally:
            for cmd in script["after"]:
                try:
                    exec_step(cmd, pty)
                except Exception as e:
                    sys.stderr.write("ERROR: {0}\n".format(e))
                    failed = True

    if failed:
        raise RuntimeError("One or more commands failed")


################################################################################
# Parsing
################################################################################

def parse_cmd(cmd, attrs):
    cmd = cmd.strip()
    if cmd.startswith("#"):
        return None
    if cmd.startswith("$"):
        cmd = cmd[1:]
    cmd = cmd.strip()
    if len(cmd) == 0:
        return None

    if "data-test-wait-for" in attrs:
        if "data-test-timeout" in attrs:
            return {"$": cmd,
                    "type": "wait",
                    "wait-for": attrs["data-test-wait-for"],
                    "timeout": attrs["data-test-timeout"]}
        else:
            return {"$": cmd, "type": "wait", "wait-for": attrs["data-test-wait-for"]}
    if "data-test-assert-contains" in attrs:
        return {"$": cmd, "type": "assert", "contains": attrs["data-test-assert-contains"]}
    if "data-test-expect" in attrs:
        return {"$": cmd, "type": "expect", "expect": attrs["data-test-expect"], "timeout": attrs["data-test-timeout"]}
    return {"$": cmd, "type": "default"}


def process_liquid(command):
    for key, value in liquid_transforms.items():
        command = re.sub(key, value, command)

    return command


def parse_cmds(pre, attrs):
    cmds = []
    line_continuation = ""
    line_continuation_delimiter = "\\"

    sanitized_cmd = process_liquid(pre)

    for line in sanitized_cmd.split("\n"):
        cmd = "{0} {1}".format(line_continuation, line.strip())
        if cmd.endswith(line_continuation_delimiter):
            line_continuation = cmd[:-len(line_continuation_delimiter)]
        else:
            cmd = parse_cmd(cmd, attrs)
            if cmd != None:
                cmds.append(cmd)
            line_continuation = ""
    return cmds


def parse_file(pre, attrs):
    if not "data-path" in attrs:
        raise ValueError("File element does not have required 'data-path' attribute.")
    path = attrs["data-path"]
    if path[0] == "/":
        raise ValueError("Absolute file paths are not permitted")
    if ".." in path:
        raise ValueError("'..' not permitted in file paths")
    content = ""
    for line in pre:
        if "ProcessingInstruction" in str(type(line)):  # xml: <? ... ?>
            content += "<?" + str(line) + ">"
        else:
            content += str(line)
    content = content[1:] if content[0] == "\n" else content
    return {"type": "file", "content": content, "path": path}


def get_macro(macro):
    if re.search("\s*init-deploy", macro):
        app_name = re.sub("\s*init-deploy\s*", "", macro).strip()
        if len(app_name) == 0:
            raise ValueError("Missing application name for macro 'init-deploy'")
        cmds = [
            {
                "$": "vespa config set target local",
                "type": "default"
            },
            {
                "$": "docker run --detach --name vespa --hostname vespa-container --publish 8080:8080 --publish 19071:19071 vespaengine/vespa",
                "type": "default"
            },
            {
                "$": "vespa status deploy --wait 300",
                "type": "default"
            },
            {
                "$": "vespa clone {} myapp && cd myapp".format(app_name),
                "type": "default"
            },
            {
                "$": "vespa deploy --wait 300 ./app",
                "type": "default"
            }
        ]
    else:
        raise ValueError("{} is not a valid macro".format(macro))

    return cmds


def parse_page(html):
    script = {
        "before": [],
        "steps": [],
        "after": []
    }

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(lambda tag: (tag.name == "pre" or tag.name == "div" or tag.name == "p") and tag.has_attr("data-test")):
        attr = tag.attrs["data-test"]

        if attr == "before":
            script["before"].extend(parse_cmds(tag.string, tag.attrs))

        if attr == "exec":
            script["steps"].extend(parse_cmds(tag.string, tag.attrs))

        if attr == "file":
            script["steps"].append(parse_file(tag.contents, tag.attrs))

        if attr == "after":
            script["after"].extend(parse_cmds(tag.string, tag.attrs))

        if re.search("^run-macro", attr):
            macro = re.sub("run-macro\s?", "", attr)
            script["steps"].extend(get_macro(macro))

    return script


def process_page(html, source_name=""):
    script = parse_page(html)

    print_cmd_header("Script to execute", extra=source_name)
    print(json.dumps(script, indent=2))

    exec_script(script)


################################################################################
# Running
################################################################################

def create_work_dir():
    os.chdir(project_root)
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)


def run_url(url):
    print_cmd_header("Testing", url)
    allpages = b""
    for page in url.split(","):
        page = page.strip()
        if page.startswith("http"):
            allpages += urllib.request.urlopen(page).read()
        else:
            with open(workdir + '/' + page, 'rb') as f:
                allpages += f.read()

    process_page(allpages, url)


def run_config(config_file):
    failed = []
    if not os.path.isfile(config_file):
        config_file = os.path.join("test", config_file)
    if not os.path.isfile(config_file):
        raise RuntimeError("Could not find configuration file")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        for url in config["urls"]:
            try:
                run_url(url)
            except RuntimeError:
                failed.append(url)

    if len(failed) > 0:
        raise RuntimeError("One or more files failed: " + ", ".join(failed))


def run_file(file_name):
    if file_name.startswith("http"):
        run_url(file_name)
    elif file_name == "-":
        process_page(sys.stdin.read(), "stdin")
    else:
        with io.open(file_name, 'r', encoding="utf-8") as f:
            process_page(f.read(), file_name)


def run_with_arguments():
    global verbose
    global workdir
    config_file = ""
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "vc:w:")
    except getopt.GetoptError:
        print("test.py [-v] [-c configfile] -w [workdir] [file-to-run]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in "-v":
            verbose = True
        elif opt in "-c":
            config_file = arg
        elif opt in "-w":
            workdir = arg

    load_liquid_transforms()

    if len(config_file):
        run_config(config_file)
    elif args:
        run_file(args[0])
    else:
        run_config("_test_config.yml")


def load_liquid_transforms():
    global liquid_transforms
    global workdir
    site_config_file = "_config.yml"
    site_config = "_config.yml"

    if not os.path.isfile(site_config):
        site_config = os.path.join("../", site_config_file)
    if not os.path.isfile(site_config):
        site_config = os.path.join(workdir, site_config_file)
    if not os.path.isfile(site_config):
        raise RuntimeError("Could not find " + site_config_file)

    # Transforms for site variables like {{site.variables.vespa_version}}
    with open(site_config, "r") as f:
        config = yaml.safe_load(f)
        if "variables" in config:
            for key, value in config["variables"].items():
                liquid_transforms[r"{{\s*site.variables."+key+r"\s*}}"] = value

    # Remove liquid macros, e.g.:
    # {% highlight shell %}       {% endhighlight %}
    # {% raw %}                   {% endraw %}
    liquid_transforms[r"{%\s*.*highlight\s*.*%}"] = ""
    liquid_transforms[r"{%\s*.*raw\s*%}"] = ""


def main():
    create_work_dir()

    try:
        run_with_arguments()
    except Exception as e:
        sys.stderr.write("ERROR: {0}\n".format(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
