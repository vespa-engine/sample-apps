const input = document.getElementById("input-field");
const output = document.getElementById("output-wrapper");

// https://www.freecodecamp.org/news/javascript-debounce-example/
const debounce = (func, timeout = 300) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

const handleResults = (data) => {
  output.innerHTML = "";

  if (data.root.children) {
    const items = data.root.children
      .map(child => ({
        title: (
          child.fields.summaryfeatures["nativeRank(title)"]*10.0 >= child.fields.summaryfeatures["nativeRank(gram_title)"]
          ? child.fields.title
          : child.fields.gram_title),
        content: (
          child.fields.summaryfeatures["nativeRank(content)"]*10.0 >= child.fields.summaryfeatures["nativeRank(gram_content)"]
          ? child.fields.content
          : child.fields.gram_content),
        path: child.fields.path
      }));

    items.forEach(item => {
      const div = document.createElement("div");
      div.className = "result";
      const title = document.createElement("h3");
      title.className = "result-title";
      title.innerHTML = item.title;
      const content = document.createElement("p");
      content.className = "result-content";
      content.innerHTML = `... ${item.content} ...`;
      const path = document.createElement("small");
      path.className = "result-path";
      path.innerHTML = `Documentation: ${item.path}`;
      div.appendChild(title);
      div.appendChild(content);
      div.appendChild(path);
      output.appendChild(div);
    });
  }
};

const handleInput = (e) => {
  if (e.target.value.length > 0) {
    const searchTerm = escape(e.target.value);
    const yqlQuery = `select+*+from+doc+where+default+contains+%22${searchTerm}%22+or+gram_title+contains+%22${searchTerm}%22+or+gram_content+contains+%22${searchTerm}%22%3B`;
    fetch(`/search/?yql=${yqlQuery}&hits=128&ranking=weighted_doc_rank&timeout=5s`)
      .then(res => res.json())
      .then(data => handleResults(data))
      .catch(e => console.error(e));
  } else {
    output.innerHTML = "";
  }
};

input.addEventListener("input", debounce(handleInput));
