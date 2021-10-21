// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

function main() {
    const input = document.getElementById("input");
    input.onkeydown = on_keydown;
    input.focus();
}

function query() {
    const input = document.getElementById("input").value;
    const url = "/search/?query=" + encodeURIComponent(input + ";");
    fetch(url)
        .then(result => result.json())
        .then(result => on_response(result))
        .catch(error => console.log(error));
}

function on_response(result) {
    console.log(result);
    const fields = result["root"]["children"][0]["fields"];
    const prediction = fields["prediction"];
    const context = fields["context"].replaceAll(prediction, "<b>" + prediction + "</b>");
    document.getElementById("answer").innerHTML = "<b>Answer:</b> " + prediction;
    document.getElementById("context").innerHTML = "<b>Passage:</b> " + context;
}

function on_keydown(e) {
    if (e.key === "Enter") {
        query();
    }
}

