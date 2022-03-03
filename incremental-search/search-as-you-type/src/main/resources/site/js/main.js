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
  const highlightWeight = 10.0;

  output.innerHTML = "";

  if (data.root.children) {
    const items = data.root.children
      .map(child => {
        const titleScore = child.fields.summaryfeatures["nativeRank(title)"];
        const gramTitleScore = child.fields.summaryfeatures["nativeRank(gram_title)"];
        const contentScore = child.fields.summaryfeatures["nativeRank(content)"];
        const gramContentScore = child.fields.summaryfeatures["nativeRank(gram_content)"];
        
        return {
          title: (
            titleScore*highlightWeight >= gramTitleScore
            ? child.fields.title
            : child.fields.gram_title),
          content: (
            contentScore*highlightWeight >= gramContentScore
            ? child.fields.content
            : child.fields.gram_content),
          path: child.fields.path
        };
      });

    items.forEach(item => {
      const div = document.createElement("div");
      div.className = "result";
      const title = document.createElement("h3");
      title.className = "result-title";
      title.innerHTML = item.title;
      const content = document.createElement("p");
      content.className = "result-content";
      content.innerHTML = item.content.replaceAll("<sep />", " ... ");
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
    const query = {
      yql: `
        select * from doc
        where ({defaultIndex: "default"}userInput(@input))
        or ({defaultIndex: "grams"}userInput(@input))`,
      input: e.target.value,
      hits: 128,
      ranking: "weighted_doc_rank",
      timeout: "5s"
    };

    fetch("/search/", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(query)
    })
      .then(res => res.json())
      .then(data => handleResults(data))
      .catch(e => console.error(e));
  } else {
    output.innerHTML = "";
  }
};

input.addEventListener("input", debounce(handleInput));
