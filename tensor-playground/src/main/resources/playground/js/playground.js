// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

contexts = {
    VIEW: "view",
    EDIT: "edit",
    SETUP: "setup"
};

var show_setup = false;
var setup = { "f": [] };
var results = { "f": [] };
var variables = new Map();
var selected = null;
var converter = new showdown.Converter();
var context = contexts.VIEW;

///////////////////////////////////////////////////////////////////////////////
// Operations and UI
///////////////////////////////////////////////////////////////////////////////

var operations = {

    ////////////////////////////////////////////////////////////
    // Comments
    "c" : {
        "params" : {
            "t" : ""  // text of comment
        },
        "setup_ui" : function(param, element, header, content, frame_index) {
            clear(header);
            header.append("div").attr("class", "block header-text").html("Edit comment");
            add_setup_ui_buttons(header.append("div").attr("class", "block right"));

            clear(content);
            add_table(content);
            add_textarea_field(content, "Comment", "", 3, 100, param["t"], true);
            add_save_cancel_field(content);
        },
        "execute" : function(param, result, next) {
            var html = converter.makeHtml(param["t"]);
            result.set("t", html);
            next();
        },
        "result_ui" : function(result, element, header, content, frame_index) {
            clear(header);
            header.append("div").attr("class", "block header-text").html("Comment");
            add_result_ui_buttons(header.append("div").attr("class", "block right"), frame_index);

            clear(content);
            content.html(result.get("t"));
        },
        "save" : function(param, element) {
            if (element.select("textarea").node() != null) {
                param["t"] = element.select("textarea").property("value").trim();
            }
        },
    },

    ////////////////////////////////////////////////////////////
    // Expressions
    "e" : {
        "params" : {
            "n" : "",  // name
            "e" : ""   // expression
        },
        "setup_ui" : function(param, element, header, content, frame_index) {
            clear(header);
            header.append("div").attr("class", "block header-text").html("Edit expression");
            add_setup_ui_buttons(header.append("div").attr("class", "block right"));

            clear(content);
            add_table(content);
            add_input_field(content, "Name", "expression_name", param["n"], true, "Only required if using in other expressions");
            add_textarea_field(content, "Expression", "expression_expression", 3, 65, param["e"], false);
            add_save_cancel_field(content);
        },
        "execute" : function(param, result, next) {
            expression = {
                "expression" : param["e"],
                "arguments" : []
            };
            variables.forEach(function (entry, name, map) {
                if (expression["expression"].includes(name)) {
                    expression["arguments"].push(entry);
                }
            });
            d3.json("/playground/eval?json=" + encodeURIComponent(JSON.stringify(expression)))
                .then(function(response) {
                    if (!has_error(response, result)) {
                        if (param["n"].length > 0) {
                            variables.set(param["n"], {
                                "name" : param["n"],
                                "type" : response["type"],
                                "value": response["type"].includes("tensor") ? response["value"]["literal"] : response["value"]
                            });
                        }
                        result.set("result", response)
                        result.set("n", param["n"])
                        result.set("type", response["type"])
                        result.set("e", param["e"])
                    }
                    next();
                });
            result.set("result", "Executing...");
            result.set("n", param["n"])
            result.set("e", param["e"])
        },
        "result_ui" : function(result, element, header, content, frame_index) {
            header.html("Expression");
            clear(content);
            if (result.size == 0) {
                content.html("Not executed yet...");
                return;
            }

            var headerLeft = "Expression";
            if (result.has("n") && result.get("n").length > 0) {
                headerLeft = "<b>" + result.get("n") + "</b>";
            }

            clear(header);
            header.append("div").attr("class", "block header-text").html(headerLeft);
            add_result_ui_buttons(header.append("div").attr("class", "block right"), frame_index);

            if (result.has("error")) {
                content.html("");
                var table = content.append("table");
                if (result.has("e") && result.get("e").length > 0) {
                    var row = table.append("tr")
                    row.append("td").attr("class", "label").html("Expression");
                    row.append("td").attr("class", "code").html(replace_html_code(result.get("e")));
                }
                var row = table.append("tr")
                row.append("td").attr("class", "label").html("Error");
                row.append("td").append("div").attr("class", "error").html(replace_html_code(result.get("error")));

            } else {
                var data = result.get("result");
                if (typeof data === "object") {
                    var table = content.append("table");

                    // expression
                    if (result.has("e") && result.get("e").length > 0) {
                        var row = table.append("tr")
                        row.append("td").attr("class", "label").html("Expression");
                        row.append("td").attr("class", "code").html(replace_html_code(result.get("e")));
                    }

                    // type
                    if (result.has("type") && result.get("type").length > 0) {
                        var row = table.append("tr")
                        row.append("td").attr("class", "label").html("Type");
                        row.append("td").attr("class", "code").html(replace_html_code(result.get("type")));
                    }

                    // value
                    var row = table.append("tr");
                    row.append("td").attr("class", "label").attr("style", "padding-top: 5px").html("Value");
                    var cell = row.append("td");
                    var value = data["value"];
                    if (data["type"] !== null && data["type"].includes("tensor")) {
                        value = data["value"]["literal"];
                    }
                    cell.append("input").attr("value", value).attr("style", "width: 800px");
                    row.append("td").append("a").attr("href", "#").attr("class", "header").html(icon_clipboard_copy())
                        .on("click", function(event) { copy_to_clipboard(value); event.stopPropagation(); event.preventDefault(); });

                    // graphical view of tensor
                    row = table.append("tr");
                    row.append("td").html("");
                    cell = row.append("td");
                    draw_table(cell, data);

                } else {
                    content.html(data);
                }
            }

        },
        "save" : function(param, element) {
            if (element.select(".expression_name").node() != null) {
                param["n"] = get_input_field_value(element, "expression_name");
                param["e"] = get_textarea_field_value(element, "expression_expression");
            }
        },
    }
}

function replace_html_code(str) {
    return str.replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function has_error(response, result) {
    result.delete("error");
    if (response == null) {
        result.set("error", "Did not receive a response.")
        return true;
    } else if ("error" in response) {
        result.set("error", response["error"])
        return true;
    }
    return false;
}

function clear(root) {
    root.html("");
}

function add_table(root) {
    root.append("table");
}

function add_label_field(root, label) {
    root = root.select("table");
    var field = root.append("tr");
    field.append("td")
        .attr("colspan", "3")
        .attr("class", "header")
        .html(label);
}

function add_input_field(root, label, classname, value, focus, helptext) {
    root = root.select("table");
    var field = root.append("tr");
    field.append("td").html(label);
    field.append("td")
        .attr("class", classname)
        .append("input")
            .attr("value", value)
            .attr("size", "50");
    field.append("td").html("<i>" + (helptext == null ? "" : helptext) + "</i>");
    if (focus) {
        field.select("input").node().select();
    }
}

function add_textarea_field(root, label, classname, rows, cols, value, focus, helptext) {
    root = root.select("table");
    var field = root.append("tr");
    field.append("td").html(label);
    field.append("td")
        .attr("class", classname)
        .append("textarea")
            .attr("rows", rows)
            .attr("cols", cols)
            .text(value);
    field.append("td").html("<i>" + (helptext == null ? "" : helptext) + "</i>");
    if (focus) {
        field.select("textarea").node().select();
    }
}

function add_save_cancel_field(root) {
    var row = root.select("table").append("tr");
    row.append("td").attr("class", "label");
    var cell = row.append("td");
    cell.append("a").attr("href", "#").html(icon_check() + " Save and execute (ctrl + enter)")
        .on("click", function(event) { execute_selected(); event.preventDefault(); });
    cell.append("a").attr("href", "#").attr("style","margin-left: 80px").html(icon_exit() + " Cancel (escape)")
        .on("click", function(event) { document.activeElement.blur(); exit_edit_selected(); event.preventDefault(); });
}

function add_setup_ui_buttons(root) {
    root.append("a").attr("href", "#").attr("class","header").html(icon_cancel())
        .on("click", function(event) { document.activeElement.blur(); exit_edit_selected(); event.preventDefault(); });
}

function add_result_ui_buttons(root, frame_index) {
    root.append("a").attr("href", "#").attr("class","header").html(icon_edit())
        .on("click", function(event) { edit_frame(frame_index); event.stopPropagation(); event.preventDefault(); });
    root.append("a").attr("href", "#").attr("class","header").html(icon_up())
        .on("click", function(event) { move_frame_up(frame_index); event.stopPropagation(); event.preventDefault(); });
    root.append("a").attr("href", "#").attr("class","header").html(icon_down())
        .on("click", function(event) { move_frame_down(frame_index); event.stopPropagation(); event.preventDefault(); });
    root.append("a").attr("href", "#").attr("class","header").html(icon_remove())
        .on("click", function(event) { remove_frame(frame_index); event.stopPropagation(); event.preventDefault(); });
}

function get_input_field_value(root, classname) {
    return root.select("." + classname).select("input").property("value").trim();
}

function get_textarea_field_value(root, classname) {
    return root.select("." + classname).select("textarea").property("value").trim();
}

function get_select_field_value(root, classname) {
    return root.select("." + classname).select("select").property("value").trim();
}

function draw_table(element, variable) {
    if (variable === null || typeof variable !== "object") {
        return;
    }

    var type = variable["type"];
    var columns = new Set();
    columns.add("__value__");

    var data = [ { "__value__": variable["value"]} ]
    if (type.includes("tensor")) {
        data = variable["value"]["cells"].map(function(cell) {
            var entry = new Object();
            var address = cell["address"];
            for (var dim in address) {
                entry[dim] = address[dim];
                columns.add(dim);
            }
            entry["__value__"] = cell["value"];
            return entry;
        });
    }

    columns = [...columns]; // sort "value" to back
    columns.sort(function(a, b) {
        var _a = a.toLowerCase(); // ignore upper and lowercase
        var _b = b.toLowerCase(); // ignore upper and lowercase
        if (_a.startsWith("__value__") && !_b.startsWith("__value__")) {
            return 1;
        }
        if (_b.startsWith("__value__") && !_a.startsWith("__value__")) {
            return -1;
        }
        if (_a < _b) {
            return -1;
        }
        if (_a > _b) {
            return 1;
        }
        return 0;
    });

    if (data.length > 25) {
        empty_row = columns.reduce((accumulator,current)=>(accumulator[current]='',accumulator), {})
        empty_row.__value__ = "...";
        filtered_data = data.filter(function(d,i) { return i < 10; });
        filtered_data = filtered_data.concat([empty_row]);
        data = filtered_data.concat(data.filter(function(d,i) { return i > data.length - 10 - 1; }));
    }
    table_html(element, data, columns);
}

function table_html(element, data, columns) {
    var table = element.append("table"),
        thead = table.append("thead"),
        tbody = table.append("tbody");

    thead.append("tr")
        .selectAll("th")
        .data(columns)
        .enter()
        .append("th")
        .text(function(column) { return column.startsWith("__value__") ? "value" : column; });

    var rows = tbody.selectAll("tr")
        .data(data)
        .enter()
        .append("tr");

    var cells = rows.selectAll("td")
        .data(function(row) {
            return columns.map(function(column) {
                return {column: column, value: row[column]};
            });
        })
        .enter()
        .append("td")
            .classed("data", true)
            .text(function(d) { return d.value; });

    return table;
}

// icons from https://systemuicons.com/
function icon_edit() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(3 3)"><path d="m7 1.5h-4.5c-1.1045695 0-2 .8954305-2 2v9.0003682c0 1.1045695.8954305 2 2 2h10c1.1045695 0 2-.8954305 2-2v-4.5003682"/><path d="m14.5.46667982c.5549155.5734054.5474396 1.48588056-.0167966 2.05011677l-6.9832034 6.98320341-3 1 1-3 6.9874295-7.04563515c.5136195-.5178979 1.3296676-.55351813 1.8848509-.1045243z"/><path d="m12.5 2.5.953 1"/></g></svg>';
}

function icon_up() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(3 3)"><path d="m3.5 7.5 4-4 4 4"/><path d="m7.5 3.5v11"/><path d="m.5.5h14"/></g></svg>';
}

function icon_down() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(3 3)"><path d="m3.5 7.5 4 4 4-4"/><path d="m7.5.5v11"/><path d="m.5 14.5h14"/></g></svg>';
}

function icon_remove() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="m7.5 7.5 6 6"/><path d="m13.5 7.5-6 6"/></g></svg>';
}

function icon_cancel() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="matrix(-1 0 0 1 18 3)"><path d="m10.595 10.5 2.905-3-2.905-3"/><path d="m13.5 7.5h-9"/><path d="m10.5.5-8 .00224609c-1.1043501.00087167-1.9994384.89621131-2 2.00056153v9.99438478c.0005616 1.1043502.8956499 1.9996898 2 2.0005615l8 .0022461"/></g></svg>';
}

function icon_clipboard_copy() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(4 3)"><path d="m6.5 11.5-3-3 3-3"/><path d="m3.5 8.5h11"/><path d="m12.5 6.5v-4.00491374c0-.51283735-.3860402-.93550867-.8833789-.99327378l-.1190802-.00672622-1.9975409.00491374m-6 0-1.99754087-.00492752c-.51283429-.00124584-.93645365.38375378-.99544161.88094891l-.00701752.11906329v10.99753792c.00061497.5520447.44795562.9996604 1 1.0006148l10 .0061554c.5128356.0008784.9357441-.3848611.993815-.8821612l.006185-.1172316v-2.5"/><path d="m4.5.5h4c.55228475 0 1 .44771525 1 1s-.44771525 1-1 1h-4c-.55228475 0-1-.44771525-1-1s.44771525-1 1-1z"/></g></svg>';
}

function icon_check() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><path d="m.5 5.5 3 3 8.028-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(5 6)"/></svg>';
}

function icon_exit() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="matrix(-1 0 0 1 18 3)"><path d="m10.595 10.5 2.905-3-2.905-3"/><path d="m13.5 7.5h-9"/><path d="m10.5.5-8 .00224609c-1.1043501.00087167-1.9994384.89621131-2 2.00056153v9.99438478c.0005616 1.1043502.8956499 1.9996898 2 2.0005615l8 .0022461"/></g></svg>';
}

function icon_comment() {
    return '<svg height="21" viewBox="0 0 21 21" width="21" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd" transform="translate(2 3)"><path d="m14.5.5c1.1045695 0 2 .8954305 2 2v10c0 1.1045695-.8954305 2-2 2l-2.999-.001-2.29389322 2.2938932c-.36048396.360484-.92771502.3882135-1.32000622.0831886l-.09420734-.0831886-2.29389322-2.2938932-2.999.001c-1.1045695 0-2-.8954305-2-2v-10c0-1.1045695.8954305-2 2-2z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/><path d="m13.5 5.5h-6" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/><path d="m4.49884033 6.5c.5 0 1-.5 1-1s-.5-1-1-1-.99884033.5-.99884033 1 .49884033 1 .99884033 1zm0 4c.5 0 1-.5 1-1s-.5-1-1-1-.99884033.5-.99884033 1 .49884033 1 .99884033 1z" fill="currentColor"/><path d="m13.5 9.5h-6" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/></g></svg>';
}

function icon_code() {
    return '<svg xmlns="http://www.w3.org/2000/svg" width="21" height="21" viewBox="0 0 21 21"><g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" transform="translate(2 3)"><line x1="10.5" x2="6.5" y1=".5" y2="14.5"/><polyline points="7.328 2.672 7.328 8.328 1.672 8.328" transform="rotate(135 4.5 5.5)"/><polyline points="15.328 6.672 15.328 12.328 9.672 12.328" transform="scale(1 -1) rotate(-45 -10.435 0)"/></g></svg>';
}

function setup_commands() {
    d3.select("#view-setup-cmd").on("click", function(event) { toggle_show_setup(); event.preventDefault(); });
    d3.select("#clear-cmd").on("click", function(event) { clear_all(); event.preventDefault(); });
    d3.select("#copy-url-cmd").on("click", function(event) { copy_to_clipboard(window.location); event.preventDefault(); });
    d3.select("#new-comment-cmd").html(icon_comment() + " " + d3.select("#new-comment-cmd").html())
        .on("click", function(event) {
            select_frame_by_index(num_frames() - 1);
            new_frame("c");
            event.preventDefault();
        });
    d3.select("#new-expression-cmd").html(icon_code() + " " + d3.select("#new-expression-cmd").html())
        .on("click", function(event) {
            select_frame_by_index(num_frames() - 1);
            new_frame("e");
            event.preventDefault();
        });
    d3.select("#apply-setup-cmd").html(icon_check() + " " + d3.select("#apply-setup-cmd").html())
        .on("click", function(event) { apply_setup(); event.preventDefault(); });
    d3.select("#close-setup-cmd").html(icon_exit() + " " + d3.select("#close-setup-cmd").html())
        .on("click", function(event) { toggle_show_setup(); event.preventDefault(); });
}

///////////////////////////////////////////////////////////////////////////////
// Setup handling
///////////////////////////////////////////////////////////////////////////////

function load_setup() {
    if (window.location.hash) {
        var compressed = window.location.hash.substring(1);
        var decompressed = LZString.decompressFromEncodedURIComponent(compressed);
        setup = JSON.parse(decompressed);
        d3.select("#setup-content").text(JSON.stringify(setup, null, 2));
        d3.select("#setup-input").attr("value", compressed);
    }
}

function on_setup_input_change() {
    var compressed = d3.select("#setup-input").property("value").trim();
    var decompressed = LZString.decompressFromEncodedURIComponent(compressed);
    setup = JSON.parse(decompressed);
    d3.select("#setup-content").text(JSON.stringify(setup, null, 2));
    save_setup();
    clear_results();
    execute_frame(0);
    document.activeElement.blur();
}

function apply_setup() {
    var setup_string = d3.select("#setup-content").property("value");
    setup = JSON.parse(setup_string);
    save_setup();
    clear_results();
    execute_frame(0);
    toggle_show_setup();
}

function save_setup() {
    var setup_string = JSON.stringify(setup, null, 2);
    d3.select("#setup-content").text(setup_string);
    var compressed = LZString.compressToEncodedURIComponent(setup_string);
    window.location.hash = compressed;
    d3.select("#setup-input").attr("value", compressed);
}

function save_changes() {
    d3.selectAll(".setup").each(function (d,i) {
        var element = d3.select(this);
        var op = d["op"];
        var param = d["p"];
        operations[op]["save"](param, element);
    });
    save_setup();
}

function clear_results() {
    results["f"] = [];
    for (var i = 0; i < setup["f"].length; ++i) {
        results["f"][i] = new Map();
    }
}

function clear_all() {
    setup = { "f": [] };
    save_setup();
    clear_results();
    update();
}

function toggle_show_setup() {
    show_setup = !show_setup;
    d3.select("#setup-container").classed("hidden", !show_setup);
    d3.select("#frames").classed("hidden", show_setup);
    d3.select("#add_frames").classed("hidden", show_setup);
    if (show_setup) {
        d3.select("#setup-content").node().focus();
        context = contexts.SETUP;
    } else {
        context = contexts.VIEW;
    }
}

function num_frames() {
    return setup["f"].length;
}

function swap(frame1, frame2) {
    var setup_frame_1 = setup["f"][frame1];
    setup["f"][frame1] = setup["f"][frame2];
    setup["f"][frame2] = setup_frame_1;

    var result_frame_1 = results["f"][frame1];
    results["f"][frame1] = results["f"][frame2];
    results["f"][frame2] = result_frame_1;
}

function remove(frame) {
    setup["f"].splice(frame, 1);
    results["f"].splice(frame, 1);
}


///////////////////////////////////////////////////////////////////////////////
// UI handling
///////////////////////////////////////////////////////////////////////////////

function new_frame(operation) {
    var default_params = JSON.stringify(operations[operation]["params"]);
    setup["f"].push({
        "op" : operation,
        "p" : JSON.parse(default_params)
    });
    results["f"].push(new Map());

    var insert_as_index = find_selected_frame_index() + 1;
    var current_index = num_frames() - 1;
    if (current_index > 0) {
        while (current_index > insert_as_index) {
            swap(current_index, current_index - 1);
            current_index -= 1;
        }
    }

    save_setup();
    update();
    select_frame_by_index(insert_as_index);
    document.activeElement.blur();
    edit_selected();
}

function update() {
    var all_data = d3.zip(setup["f"], results["f"]);

    var rows = d3.select("#frames").selectAll(".frame").data(all_data);
    rows.exit().remove();
    var frames = rows.enter()
        .append("div")
            .on("click", function() { select_frame(this); })
            .attr("class", "frame");
    frames.append("div").attr("class", "frame-header").html("header");
    frames.append("div").attr("class", "frame-content").html("content");

    d3.select("#frames").selectAll(".frame").data(all_data).each(function (d, i) {
        var element = d3.select(this);
        var op = d[0]["op"];
        var param = d[0]["p"];
        var result = d[1];

        var header = element.select(".frame-header");
        var content = element.select(".frame-content");

        operations[op]["result_ui"](result, element, header, content, i);
     });
}

function remove_frame(frame_index) {
    select_frame_by_index(frame_index);
    remove_selected();
}

function remove_selected() {
    var frame = find_selected_frame_index();
    remove(frame);
    save_setup();
    update();
    select_frame_by_index(frame);
}

function move_frame_up(frame_index) {
    select_frame_by_index(frame_index);
    move_selected_up();
}

function move_selected_up() {
    frame_index = find_selected_frame_index();
    if (frame_index == 0) {
        return;
    }
    swap(frame_index, frame_index-1);
    save_setup();
    update();
    select_frame_by_index(frame_index-1);
}

function move_frame_down(frame_index) {
    select_frame_by_index(frame_index);
    move_selected_down();
}

function move_selected_down() {
    frame_index = find_selected_frame_index();
    if (frame_index == setup["f"].length - 1) {
        return;
    }
    swap(frame_index, frame_index+1);
    save_setup();
    update();
    select_frame_by_index(frame_index+1);
}

function execute_selected() {
    var frame = d3.select(selected);
    var data = frame.data();
    var setup = data[0][0]; // because of zip in update
    var op = setup["op"];
    var param = setup["p"];

    operations[op]["save"](param, frame);
    save_setup();

    execute_frame(find_selected_frame_index());
    exit_edit_selected();
}

function execute_frame(i) {
    if (i < 0) {
        return;
    }
    if (i >= setup["f"].length) {
        update();
        return;
    }
    var op = setup["f"][i]["op"];
    var params = setup["f"][i]["p"];
    var result = results["f"][i];
    operations[op]["execute"](params, result, function(){ execute_frame(i+1); });
}

function find_selected_frame_index() {
    var result = -1;
    d3.select("#frames").selectAll(".frame")
        .each(function (d, i) {
            if (this.classList.contains("selected")) {
                result = i;
            }
        });
    return result;
}

function find_frame_index(frame) {
    var result = null;
    d3.select("#frames").selectAll(".frame")
        .each(function (d, i) {
            if (this == frame) {
                result = i;
            }
        });
    return result;
}

function is_element_entirely_visible(el) {
    var rect = el.getBoundingClientRect();
    var height = window.innerHeight || doc.documentElement.clientHeight;
    return !(rect.top < 50 || rect.bottom > height);
}

function select_frame(frame) {
    if (selected == frame) {
        return;
    }
    if (context === contexts.EDIT) {
        exit_edit_selected();
    }
    if (selected != null) {
        selected.classList.remove("selected");
    }
    selected = frame;
    selected.classList.add("selected");
    if (!is_element_entirely_visible(selected)) {
        selected.scrollIntoView();
        document.scrollingElement.scrollTop -= 60;
    }
    selected_frame_index = find_selected_frame_index();
}

function select_frame_by_index(i) {
    if (i >= num_frames()) {
        i = num_frames() - 1;
    }
    if (i < 0) {
        i = 0;
    }
    d3.select("#frames").selectAll(".frame")
        .each(function (datum, index) {
            if (i == index) {
                select_frame(this);
            }
        });
}

function edit_frame(frame_index) {
    select_frame_by_index(frame_index);
    edit_selected();
}

function edit_selected() {
    if (context === contexts.EDIT) {
        exit_edit_selected();
        return;
    }
    var frame = d3.select(selected);
    var data = frame.data();
    var setup = data[0][0]; // because of zip in update
    var result = data[0][1];

    var op = setup["op"];
    var param = setup["p"];

    var header = frame.select(".frame-header");
    var content = frame.select(".frame-content");

    operations[op]["setup_ui"](param, frame, header, content, find_frame_index(frame));

    context = contexts.EDIT;
}

function exit_edit_selected() {
    if (context !== contexts.EDIT) {
        return;
    }

    var frame = d3.select(selected);
    var data = frame.data();
    var setup = data[0][0]; // because of zip in update
    var result = data[0][1];

    var op = setup["op"];
    var param = setup["p"];

    var header = frame.select(".frame-header");
    var content = frame.select(".frame-content");

    operations[op]["result_ui"](result, frame, header, content, find_frame_index(frame));

    context = contexts.VIEW;
}

function event_in_input(event) {
    var tag_name = d3.select(event.target).node().tagName;
    return (tag_name == 'INPUT' || tag_name == 'SELECT' || tag_name == 'TEXTAREA' || tag_name == 'BUTTON');
}

function event_in_frame(event) {
    var node = d3.select(event.target).node();
    while (node != null) {
        if (d3.select(node).attr("class") != null) {
            if (d3.select(node).attr("class").includes("frame")) {
                return true
            }
        }
        node = node.parentElement;
    }
    return false;
}

function copy_to_clipboard(text) {
    var textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
}

function setup_keybinds() {
    var previous_keydown = { "key" : null, "ts" : 0 };

    key_binds = {}
    key_binds[contexts.VIEW] = {}
    key_binds[contexts.EDIT] = {}
    key_binds[contexts.SETUP] = {}

    key_binds[contexts.VIEW]["up"] =
    key_binds[contexts.VIEW]["k"]  = function() { select_frame_by_index(find_selected_frame_index() - 1); };
    key_binds[contexts.VIEW]["down"] =
    key_binds[contexts.VIEW]["j"]    = function() { select_frame_by_index(find_selected_frame_index() + 1); };

    key_binds[contexts.VIEW]["shift + up"] =
    key_binds[contexts.VIEW]["shift + k"]  = function() { move_selected_up(); };
    key_binds[contexts.VIEW]["shift + down"] =
    key_binds[contexts.VIEW]["shift + j"]  = function() { move_selected_down(); };

    key_binds[contexts.VIEW]["backspace"] =
    key_binds[contexts.VIEW]["x"]   = function() { remove_selected(); };
    key_binds[contexts.VIEW]["d,d"]   = function() { remove_selected(); };

    key_binds[contexts.VIEW]["e"]  =
    key_binds[contexts.VIEW]["enter"]   = function() { edit_selected(); };
    key_binds[contexts.VIEW]["ctrl + enter"]   = function() { execute_selected(); };

    key_binds[contexts.VIEW]["n,c"] = function() { new_frame("c"); };
    key_binds[contexts.VIEW]["n,e"] = function() { new_frame("e"); };
    key_binds[contexts.VIEW]["a"]   = function() { new_frame(null); };

    key_binds[contexts.VIEW]["esc"] = function() { document.activeElement.blur(); };
    key_binds[contexts.SETUP]["esc"] = function() { document.activeElement.blur(); };

    key_binds[contexts.EDIT]["esc"] = function() { document.activeElement.blur(); exit_edit_selected(); };
    key_binds[contexts.EDIT]["ctrl + enter"] = function() { execute_selected(); };


    d3.select('body').on('keydown', function(event) {
        var combo = [];

        if (event.shiftKey) combo.push("shift");
        if (event.ctrlKey) combo.push("ctrl");
        if (event.altKey) combo.push("alt");
        if (event.metaKey) combo.push("meta");

        var key_code = event.keyCode;

        if (key_code == 8) combo.push("backspace");
        if (key_code == 13) combo.push("enter");
        if (key_code == 27) combo.push("esc");
        if (key_code == 32) combo.push("space");
        if (key_code == 46) combo.push("del");

        if (key_code == 37) combo.push("left");
        if (key_code == 38) combo.push("up");
        if (key_code == 39) combo.push("right");
        if (key_code == 40) combo.push("down");

        // a-z
        if (key_code >= 64 && key_code < 91) combo.push(String.fromCharCode(key_code).toLowerCase());

        var key = combo.join(" + ");
        if (event_in_input(event) && !event_in_frame(event) && key !== "esc") {
            return;
        }

        // Check if combo combined with previous key is bound
        if (Date.now() - previous_keydown["ts"] < 400) {
            var two_key = previous_keydown["key"] + "," + key;
            if (two_key in key_binds[context]) {
                key_binds[context][two_key]();
                event.preventDefault();
                previous_keydown = { "key":null, "ts": 0 };  // reset
                return;
            }
        }

        if (key in key_binds[context]) {
            key_binds[context][key]();
            event.preventDefault();
        }

        previous_keydown = { "key":key, "ts": Date.now() };
    });
}

function main() {
    setup_commands();
    load_setup();
    clear_results();
    execute_frame(0);
    update();
    select_frame_by_index(0);
    setup_keybinds();
}

