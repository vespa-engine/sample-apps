const input = document.getElementById("input-field");
const output = document.getElementById("output-wrapper");

const dropdown = document.getElementById("results");

// https://www.freecodecamp.org/news/javascript-debounce-example/
const debounce = (func, timeout = 300) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      func.apply(this, args);
    }, timeout);
  };
};

const hideDropdown = () => {
  output.innerHTML = "";
  dropdown.classList.remove("show");
  dropdown.classList.add("hide");
};

const handleSuggestClick = (e) => {
  e.preventDefault();
  e.stopPropagation();
  input.value = e.target.innerHTML;
};

const handleUnfocus = (e) => hideDropdown();

const handleResults = (data) => {
  dropdown.innerHTML = "";

  if (data.root.children) {
    const items = data.root.children.map((child) => ({
      term: child.fields.term,
    }));
    items.forEach((item) => {
      const p = document.createElement("p");
      p.innerHTML = item.term;
      p.addEventListener("mousedown", handleSuggestClick);
      dropdown.appendChild(p);
    });
  }
};

const handleInput = (e) => {
  dropdown.innerHTML = "";
  if (e.target.value.length > 0) {
    dropdown.classList.add("show");
    dropdown.classList.remove("hide");

    const query = {
      yql: `
        select * from term
        where default contains ({prefix:true} "${e.target.value.replaceAll(/[^a-zA-Z0-9 ]/g, "")}")`,
      ranking: "term_rank",
      hits: 10,
      timeout: "5s",
    };

    fetch("/search/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(query),
    })
      .then((res) => res.json())
      .then(handleResults)
      .catch(console.error);
  } else {
    hideDropdown();
  }
};

input.addEventListener("input", debounce(handleInput));
input.addEventListener("focusout", handleUnfocus);
