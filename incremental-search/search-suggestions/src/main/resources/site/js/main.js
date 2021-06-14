const input = document.getElementById("input-field");
const output = document.getElementById("output-wrapper");


const dropdown = document.getElementById("results")

// https://www.freecodecamp.org/news/javascript-debounce-example/
const debounce = (func, timeout = 300) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

const handleResults = (data) => {
  console.log(data);
  output.innerHTML = "";

  if (data.root.children[0].children) {
    const items = data.root.children[0].children[0].children
      .map(child => ({
        value: (child.value)
      }));
  
    console.log(items);
    


    items.map(item => {
      const p = document.createElement("p");
      p.innerHTML = item.value;
      p.addEventListener("click",(e) => input.value=e.target.innerHTML);
      dropdown.appendChild(p)
    });
    
  }
};

const handleInput = (e) => {
  dropdown.innerHTML = "";
  if (e.target.value.length > 0) {
    dropdown.classList.add("show");
    dropdown.classList.remove("hide");
    const searchTerm = escape(e.target.value);
    
    const query = {
      yql: `
        select * from query where ([{"defaultIndex": "default"}]userInput(@input)) | all(group(input) max(10) order(-avg(relevance()) * count())  each(max(1)));`,
      input: e.target.value,
      timeout: "5s"
    };
   
    fetch("/search/?streaming.groupname=0", {
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
    dropdown.classList.remove("show");
    dropdown.classList.add("hide");
  }
};

input.addEventListener("input", debounce(handleInput));
