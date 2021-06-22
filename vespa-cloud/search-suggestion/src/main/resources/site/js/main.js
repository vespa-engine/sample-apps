const input = document.getElementById("input-field");
const output = document.getElementById("output-wrapper");

const dropdown = document.getElementById("results");
const termDropdown = document.getElementById("termResults");

// https://www.freecodecamp.org/news/javascript-debounce-example/
const debounce = (func, timeout = 300) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

const hideDropdown = () => {
  output.innerHTML = "";
  dropdown.classList.remove("show");
  dropdown.classList.add("hide");
  termDropdown.classList.remove("show");
  termDropdown.classList.add("hide");
};

const handleSuggestClick = (e) => {
  e.preventDefault();
  e.stopPropagation();
  input.value = e.target.innerHTML;
};

const handleUnfocus = (e) => hideDropdown();

const handleResults = (data) => {
  output.innerHTML = "";
  const small = document.createElement("small");
  small.innerHTML = "Query log queries";
  dropdown.appendChild(small);

  if (data.root.children[0].children) {
    const items = data.root.children[0].children[0].children
      .map(child => ({
        value: (child.value)
      }));

    items.map(item => {
      const p = document.createElement("p");
      p.innerHTML = item.value;
      p.addEventListener("mousedown", handleSuggestClick);
      dropdown.appendChild(p)
    });
    
  }
};

const handleTermResults = (data) => {
  termDropdown.innerHTML = "";
  termDropdown.appendChild(document.createElement("hr"));
  const small = document.createElement("small");
  small.innerHTML = "Bootstrapped search terms";
  termDropdown.appendChild(small);

  if (data.root.children) {
    const items = data.root.children
      .map(child => ({
        term: child.fields.term
      }));

    items.forEach(item => {
      const p = document.createElement("p");
      p.innerHTML = item.term;
      p.addEventListener("mousedown", handleSuggestClick);
      termDropdown.appendChild(p)
    });
  }
};

const handleInput = (e) => {
  dropdown.innerHTML = "";
  if (e.target.value.length > 0) {
    dropdown.classList.add("show");
    dropdown.classList.remove("hide");
    termDropdown.classList.add("show");
    termDropdown.classList.remove("hide");
    
    const queryQuery = {
      yql: `
        select * from query where ([{"defaultIndex": "default"}]userInput(@input)) | all(group(input) max(10) order(-avg(relevance()) * count())  each(max(1)));`,
      hits: 10,
      input: e.target.value,
      timeout: "5s"
    };
   
    fetch("/search/?streaming.groupname=0", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(queryQuery)
    })
      .then(res => res.json())
      .then(data => handleResults(data))
      .catch(e => console.error(e));

    const termQuery = {
      yql: `
        select * from term where ([{"defaultIndex": "default"}]userInput(@input));`,
      ranking: "term_rank",
      hits: 10,
      input: e.target.value,
      timeout: "5s"
    };
   
    fetch("/search/?streaming.groupname=0", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(termQuery)
    })
      .then(res => res.json())
      .then(data => handleTermResults(data))
      .catch(e => console.error(e));
    
  } else {
    hideDropdown();
  }
};

input.addEventListener("input", debounce(handleInput));
input.addEventListener("focusout", handleUnfocus);
