const suggestions = ['gerber knifes', 'gerber sharpeners', 'stuff'];

document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('searchtext');
  const suggestionList = document.getElementById('suggestions');
  let selectedSuggestionIndex = -1;

  searchInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      const selectedSuggestion = suggestionList.querySelector('.selected');
      if (selectedSuggestion) {
        searchInput.value = selectedSuggestion.textContent;
        suggestionList.innerHTML = '';
        suggestionList.style.display = 'none';
      }
      suggestionList.closest('form').submit();
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const suggestionItems = suggestionList.querySelectorAll('li');
      const numberOfSuggestions = suggestionItems.length;
      if (numberOfSuggestions) {
        let newSelectedSuggestionIndex = -1;
        if (event.key === 'ArrowDown') {
          newSelectedSuggestionIndex = (selectedSuggestionIndex + 1) % numberOfSuggestions;
        } else if (event.key === 'ArrowUp') {
          newSelectedSuggestionIndex = (selectedSuggestionIndex - 1 + numberOfSuggestions) % numberOfSuggestions;
        }
        const newSelectedSuggestion = suggestionItems[newSelectedSuggestionIndex];
        const selectedSuggestion = suggestionList.querySelector('.selected');
        if (selectedSuggestion) {
          selectedSuggestion.classList.remove('selected');
        }
        if (newSelectedSuggestion) {
          newSelectedSuggestion.classList.add('selected');
          searchInput.value = newSelectedSuggestion.textContent;
        } else {
          searchInput.value = searchInput.getAttribute('data-previous-value');
        }
        selectedSuggestionIndex = newSelectedSuggestionIndex;
      }
    }
  });

  suggestionList.addEventListener('click', function(event) {
    if (event.target.tagName === 'LI') {
      searchInput.value = event.target.textContent;
      suggestionList.innerHTML = '';
      suggestionList.style.display = 'none';
      suggestionList.closest('form').submit();
    }
  });

  searchInput.addEventListener('input', function() {
    let inputValue = searchInput.value.toLowerCase();
    inputValue = inputValue.trim()
    if(inputValue.length === 0) {
        suggestionList.innerHTML = '';
        return;
    }

    fetch(`/search/?term=${inputValue}&hits=5`).then(
        (response) => response.json()).then((data) => {
            suggestionList.innerHTML = '';
            if (data['root']['fields']['totalCount'] > 0) {
                data['root']['children'].forEach( function(hit) {
                    const suggestionItem = document.createElement('li');
                    suggestionItem.textContent = hit['fields']['query'];
                    suggestionList.appendChild(suggestionItem);
                });
                suggestionList.style.display = suggestions.length ? 'block' : 'none';
                searchInput.setAttribute('data-previous-value', searchInput.value);
                selectedSuggestionIndex = -1;
           }
        });
  });
});

