document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsContainer = document.getElementById('results-container');
    const preferencesPillContainer = document.getElementById('preferences-pill-container');
    const preferencesPills = document.getElementById('preferences-pills');
    const makeFacetsContainer = document.getElementById('make-facets').querySelector('.facet-items');
    const modelFacetsContainer = document.getElementById('model-facets').querySelector('.facet-items');
    const modelFacetsSection = document.getElementById('model-facets');
    const transmissionFacetsContainer = document.getElementById('transmission-facets').querySelector('.facet-items');
    const fuelTypeFacetsContainer = document.getElementById('fuelType-facets').querySelector('.facet-items');
    
    // Templates
    const messageTemplate = document.getElementById('message-template');
    const carCardTemplate = document.getElementById('car-card-template');
    const facetItemTemplate = document.getElementById('facet-item-template');
    
    // Store manually adjusted preferences
    let manualPreferences = {};
    
    // Store selected facet filters
    let selectedFacets = {
        make: [],
        model: [],
        transmission: [],
        fuelType: []
    };
    
    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    resetBtn.addEventListener('click', resetConversation);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Initialize with a default search
    function initialize() {
        // Set a default preference for price=0
        manualPreferences = { price: 0 };
        
        // Show loading indicator
        loadingOverlay.classList.remove('d-none');
        
        // Display the default preference
        displayPreferences(manualPreferences);
        preferencesPillContainer.classList.remove('d-none');
        
        // Run the initial search
        fetchInitialSearch();
    }
    
    // Fetch initial search results
    async function fetchInitialSearch() {
        try {
            const response = await fetch('/api/update_preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    preferences: manualPreferences,
                    facet_filters: selectedFacets
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Display search results and facets
            if (data.search_results) {
                displaySearchResults(data.search_results, data.facet_results);
            }
        } catch (error) {
            console.error('Error fetching initial results:', error);
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Show error message in results container
            resultsContainer.innerHTML = `
            <div class="text-center p-5">
                <i class="fas fa-exclamation-triangle fa-3x mb-3 text-warning"></i>
                <h3>Error loading results</h3>
                <p>There was a problem loading the initial search results. Please try a search query.</p>
            </div>`;
        }
    }
    
    // Send message function
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true);
        userInput.value = '';
        
        // Show loading
        loadingOverlay.classList.remove('d-none');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message,
                    manual_preferences: manualPreferences,
                    facet_filters: selectedFacets,
                    is_preference_adjustment: false
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add assistant response
            addMessage(data.response, false);
            
            // Display preferences if available
            if (data.preferences) {
                // Store the new preferences in manualPreferences
                // This ensures manually adjusted values persist
                Object.keys(data.preferences).forEach(key => {
                    if (!manualPreferences.hasOwnProperty(key)) {
                        manualPreferences[key] = data.preferences[key];
                    }
                });
                displayPreferences(data.preferences);
            } else {
                displayPreferences(manualPreferences);
            }
            
            // Display search results if available
            if (data.search_results) {
                displaySearchResults(data.search_results, data.facet_results);
            } else {
                // Show initial state for car results but keep facets if they exist
                resultsContainer.innerHTML = `
                <div class="initial-state">
                    <div class="text-center p-5">
                        <i class="fas fa-car-side fa-4x mb-3"></i>
                        <h3>Looking for your perfect car?</h3>
                        <p>Tell me about your preferences, and I'll help you find matching cars.</p>
                        <p class="text-muted">For example: "I want a cheap car that's fuel efficient"</p>
                    </div>
                </div>`;
                
                // Clear facets
                makeFacetsContainer.innerHTML = '';
                modelFacetsContainer.innerHTML = '';
                transmissionFacetsContainer.innerHTML = '';
                fuelTypeFacetsContainer.innerHTML = '';
            }
        } catch (error) {
            console.error('Error:', error);
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add error message
            addMessage('Sorry, there was an error processing your request. Please try again.', false);
        }
    }
    
    // Add message to chat
    function addMessage(content, isUser) {
        // If it's an assistant message, strip out the JSON part before displaying
        let displayContent = content;
        if (!isUser) {
            const jsonIndex = displayContent.indexOf("===JSON");
            if (jsonIndex !== -1) {
                displayContent = displayContent.substring(0, jsonIndex).trim();
            }
        }
        
        const messageNode = messageTemplate.content.cloneNode(true);
        const messageDiv = messageNode.querySelector('.message');
        const messageContent = messageNode.querySelector('.message-content');
        
        messageDiv.classList.add(isUser ? 'user' : 'assistant');
        messageContent.textContent = displayContent;
        
        chatContainer.appendChild(messageNode);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Display preferences as pills
    function displayPreferences(preferences) {
        preferencesPills.innerHTML = '';
        
        if (!preferences || Object.keys(preferences).length === 0) {
            preferencesPillContainer.classList.add('d-none');
            return;
        }
        
        Object.entries(preferences).forEach(([key, value]) => {
            const pill = document.createElement('div');
            pill.className = 'preference-pill';
            pill.setAttribute('data-preference-key', key);
            
            const valueStr = value >= 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
            const label = key.charAt(0).toUpperCase() + key.slice(1);
            
            pill.innerHTML = `${label} <span class="preference-weight">${valueStr}</span>`;
            pill.addEventListener('click', () => showSliderForPreference(key, value));
            preferencesPills.appendChild(pill);
        });
        
        preferencesPillContainer.classList.remove('d-none');
    }
    
    // Show slider for preference adjustment
    function showSliderForPreference(key, currentValue) {
        // Create modal backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        document.body.appendChild(backdrop);
        
        // Create slider container
        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'preference-slider-container';
        
        const title = document.createElement('h3');
        title.textContent = `Adjust ${key.charAt(0).toUpperCase() + key.slice(1)} Preference`;
        sliderContainer.appendChild(title);
        
        // Create value display
        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'slider-value';
        valueDisplay.textContent = currentValue.toFixed(1);
        sliderContainer.appendChild(valueDisplay);
        
        // Create slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '-5';
        slider.max = '5';
        slider.step = '0.1';
        slider.value = currentValue;
        slider.className = 'preference-slider';
        
        // Update value display when slider moves
        slider.addEventListener('input', () => {
            const value = parseFloat(slider.value);
            valueDisplay.textContent = value.toFixed(1);
        });
        
        sliderContainer.appendChild(slider);
        
        // Create buttons container
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'slider-buttons';
        
        // Remove button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn btn-danger';
        removeBtn.textContent = 'Remove';
        removeBtn.addEventListener('click', () => {
            // Remove this preference
            delete manualPreferences[key];
            
            // Update UI - remove the pill
            const pill = preferencesPills.querySelector(`[data-preference-key="${key}"]`);
            if (pill) {
                preferencesPills.removeChild(pill);
            }
            
            // If no pills remain, hide the container
            if (preferencesPills.children.length === 0) {
                preferencesPillContainer.classList.add('d-none');
            }
            
            // Remove modal
            document.body.removeChild(backdrop);
            document.body.removeChild(sliderContainer);
            
            // Create a message about removing the preference
            const message = `I'd like to remove ${key} from my preferences.`;
            
            // Add user message to chat
            addMessage(message, true);
            
            // Send the preference removal message
            sendPreferenceRemoval(message, key);
        });
        buttonsContainer.appendChild(removeBtn);
        
        // Cancel button
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-secondary';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.addEventListener('click', () => {
            document.body.removeChild(backdrop);
            document.body.removeChild(sliderContainer);
        });
        buttonsContainer.appendChild(cancelBtn);
        
        // Apply button
        const applyBtn = document.createElement('button');
        applyBtn.className = 'btn btn-primary';
        applyBtn.textContent = 'Apply';
        applyBtn.addEventListener('click', () => {
            const newValue = parseFloat(slider.value);
            const oldValue = manualPreferences[key] || 0;
            manualPreferences[key] = newValue;
            
            // Update the pill display
            const pill = preferencesPills.querySelector(`[data-preference-key="${key}"]`);
            if (pill) {
                const valueSpan = pill.querySelector('.preference-weight');
                const valueStr = newValue >= 0 ? `+${newValue.toFixed(1)}` : newValue.toFixed(1);
                valueSpan.textContent = valueStr;
            }
            
            // Remove modal
            document.body.removeChild(backdrop);
            document.body.removeChild(sliderContainer);
            
            // Create a message about the preference change
            const direction = newValue > oldValue ? "increase" : "decrease";
            const message = `I'd like to ${direction} the importance of ${key} from ${oldValue.toFixed(1)} to ${newValue.toFixed(1)}.`;
            
            // Add user message to chat
            addMessage(message, true);
            
            // Send the preference adjustment message
            sendPreferenceAdjustment(message, key, newValue);
        });
        buttonsContainer.appendChild(applyBtn);
        
        sliderContainer.appendChild(buttonsContainer);
        document.body.appendChild(sliderContainer);
    }
    
    // Send preference adjustment message
    async function sendPreferenceAdjustment(message, key, value) {
        // Show loading
        loadingOverlay.classList.remove('d-none');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message,
                    is_preference_adjustment: true,
                    adjustment: {
                        key: key,
                        value: value
                    },
                    manual_preferences: manualPreferences,
                    facet_filters: selectedFacets
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add assistant response
            addMessage(data.response, false);
            
            // Display preferences if available
            if (data.preferences) {
                displayPreferences(data.preferences);
            } else {
                displayPreferences(manualPreferences);
            }
            
            // Display search results if available
            if (data.search_results) {
                displaySearchResults(data.search_results, data.facet_results);
            }
        } catch (error) {
            console.error('Error updating preferences:', error);
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add error message
            addMessage('Sorry, there was an error updating your preferences. Please try again.', false);
        }
    }
    
    // Send preference removal message
    async function sendPreferenceRemoval(message, keyToRemove) {
        // Show loading
        loadingOverlay.classList.remove('d-none');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message,
                    is_preference_adjustment: true,
                    is_preference_removal: true,
                    removal: {
                        key: keyToRemove
                    },
                    manual_preferences: manualPreferences,
                    facet_filters: selectedFacets
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add assistant response
            addMessage(data.response, false);
            
            // Display preferences if available
            if (data.preferences) {
                displayPreferences(data.preferences);
            }
            
            // Display search results if available
            if (data.search_results) {
                displaySearchResults(data.search_results, data.facet_results);
            }
        } catch (error) {
            console.error('Error removing preference:', error);
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add error message
            addMessage('Sorry, there was an error removing your preference. Please try again.', false);
        }
    }
    
    // Update search results with current preferences
    async function updateSearchResults() {
        // Show loading
        loadingOverlay.classList.remove('d-none');
        
        try {
            const response = await fetch('/api/update_preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    preferences: manualPreferences,
                    facet_filters: selectedFacets
                }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Display updated search results
            if (data.search_results) {
                displaySearchResults(data.search_results, data.facet_results);
            }
        } catch (error) {
            console.error('Error updating preferences:', error);
            
            // Hide loading
            loadingOverlay.classList.add('d-none');
            
            // Add error message
            addMessage('Sorry, there was an error updating your preferences. Please try again.', false);
        }
    }
    
    // Display search results
    function displaySearchResults(results, facetResults) {
        // Clear previous results
        resultsContainer.innerHTML = '';
        
        // Display facets if available
        if (facetResults) {
            displayFacets(parseFacetResults(facetResults));
        }
        
        if (!results || !results.root || !results.root.children || results.root.children.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">No cars found matching your criteria.</div>';
            // Reset scroll position to top
            resultsContainer.scrollTop = 0;
            return;
        }
        
        // Display each car
        results.root.children.forEach(car => {
            const carNode = carCardTemplate.content.cloneNode(true);
            const carCard = carNode.querySelector('.car-card');
            
            // Set the car title
            const title = carCard.querySelector('.car-title');
            title.textContent = `${car.fields.make} ${car.fields.model}`;
            
            // Set the car image
            const imageContainer = carCard.querySelector('.car-image');
            const carImage = carCard.querySelector('.car-img');
            
            // Add loading class to image container
            imageContainer.classList.add('loading');
            
            // Set a default placeholder initially
            carImage.src = 'https://via.placeholder.com/400x200?text=Loading...';
            
            // Load image asynchronously
            if (car.fields.make && car.fields.model) {
                loadCarImageAsync(car.fields.make, car.fields.model, carImage, imageContainer);
            } else {
                // If missing make or model, use placeholder
                carImage.src = 'https://via.placeholder.com/400x200?text=No+Image+Available';
                imageContainer.classList.remove('loading');
            }
            
            // Set the car details
            const price = carCard.querySelector('.price');
            price.textContent = `$${car.fields.price.toLocaleString()}`;
            
            const year = carCard.querySelector('.year');
            year.textContent = car.fields.year;
            
            const mileage = carCard.querySelector('.mileage');
            mileage.textContent = `${car.fields.mileage.toLocaleString()} miles`;
            
            const transmission = carCard.querySelector('.transmission');
            transmission.textContent = car.fields.transmission;
            
            const fuelType = carCard.querySelector('.fuelType');
            fuelType.textContent = car.fields.fuelType;
            
            resultsContainer.appendChild(carNode);
        });
        
        // Reset scroll position to top after displaying new results
        resultsContainer.scrollTop = 0;
    }
    
    // Function to load car images asynchronously
    async function loadCarImageAsync(make, model, imgElement, containerElement) {
        try {
            console.log(`Loading image for ${make} ${model}...`);
            
            // Create a cache key for debugging
            const cacheKey = `${make}_${model}`.toLowerCase().replace(/\s+/g, '_');
            
            // Use a fallback image from an external API in case Wikipedia fails
            const wikimediaResponse = await fetch(`/api/car_image/${encodeURIComponent(make)}/${encodeURIComponent(model)}`);
            const wikiResult = await wikimediaResponse.json();
                
            if (wikiResult.image_url) {
                // Show cache status for debugging
                if (wikiResult.from_cache) {
                    console.log(`Using cached image for ${make} ${model}`);
                } else {
                    console.log(`Retrieved new image for ${make} ${model}`);
                }
                
                // Use Wikipedia image
                imgElement.src = wikiResult.image_url;
                imgElement.onerror = function() {
                    // If Wikipedia image fails to load, use placeholder
                    imgElement.src = 'https://via.placeholder.com/400x200?text=No+Image+Available';
                };
            } else {
                // Use fallback
                imgElement.src = 'https://via.placeholder.com/400x200?text=No+Image+Available';
            }
            
            // Remove loading class once the image has loaded
            imgElement.onload = function() {
                containerElement.classList.remove('loading');
            };
        } catch (error) {
            console.error(`Error loading image for ${make} ${model}:`, error);
            imgElement.src = 'https://via.placeholder.com/400x200?text=No+Image+Available';
            containerElement.classList.remove('loading');
        }
    }
    
    // Parse facet results from the separate facet request
    function parseFacetResults(facetResults) {
        const parsedFacets = {
            make: [],
            model: [],
            transmission: [],
            fuelType: []
        };
        
        if (!facetResults || !facetResults.root || !facetResults.root.children) {
            return parsedFacets;
        }
        
        // Extract facet data from the dedicated facet query
        facetResults.root.children.forEach(item => {
            if (item.id && item.id.startsWith('group:root')) {
                // Extract facet data
                if (item.children && item.children.length > 0) {
                    const grouplist = item.children[0];
                    if (grouplist.label) {
                        // Store facet items in the appropriate array
                        parsedFacets[grouplist.label] = grouplist.children || [];
                    }
                }
            }
        });
        
        return parsedFacets;
    }
    
    // Display facets
    function displayFacets(facetResults) {
        // Clear previous facets
        makeFacetsContainer.innerHTML = '';
        modelFacetsContainer.innerHTML = '';
        transmissionFacetsContainer.innerHTML = '';
        fuelTypeFacetsContainer.innerHTML = '';
        
        // Get max counts for each facet type to calculate relative bar widths
        const getMaxCount = items => {
            if (!items || items.length === 0) return 1; // Prevent division by zero
            return Math.max(...items.map(item => item.fields ? item.fields['count()'] || 0 : 0));
        };
        
        const maxMakeCount = getMaxCount(facetResults.make);
        const maxModelCount = getMaxCount(facetResults.model);
        const maxTransmissionCount = getMaxCount(facetResults.transmission);
        const maxFuelTypeCount = getMaxCount(facetResults.fuelType);
        
        // Helper function to render a facet item
        const renderFacetItem = (item, container, maxCount, facetType) => {
            if (!item.fields || !item.fields['count()']) return;
            
            const facetNode = facetItemTemplate.content.cloneNode(true);
            const facetItem = facetNode.querySelector('.facet-item');
            
            // Check if this facet is currently selected
            const isSelected = selectedFacets[facetType].includes(item.value);
            if (isSelected) {
                facetItem.classList.add('selected');
                facetItem.setAttribute('title', 'Click to remove this filter');
                
                // Add a remove indicator
                const removeIcon = document.createElement('span');
                removeIcon.className = 'facet-remove-icon';
                removeIcon.innerHTML = '×';
                facetItem.appendChild(removeIcon);
            } else {
                facetItem.setAttribute('title', 'Click to filter by ' + item.value);
            }
            
            // Create header div for name and count
            const header = document.createElement('div');
            header.className = 'facet-item-header';
            
            // Setup name
            const nameSpan = document.createElement('span');
            nameSpan.className = 'facet-name';
            nameSpan.textContent = item.value;
            header.appendChild(nameSpan);
            
            // Setup count
            const countSpan = document.createElement('span');
            countSpan.className = 'facet-count';
            countSpan.textContent = item.fields['count()'].toLocaleString();
            header.appendChild(countSpan);
            
            facetItem.appendChild(header);
            
            // Setup bar with relative width
            const barWidth = (item.fields['count()'] / maxCount) * 100;
            const bar = facetItem.querySelector('.facet-bar');
            bar.style.width = `${barWidth}%`;
            
            // Add click event to toggle this facet filter
            facetItem.addEventListener('click', () => {
                toggleFacetFilter(facetType, item.value);
            });
            
            container.appendChild(facetNode);
        };
        
        // Render each facet type
        facetResults.make.forEach(item => renderFacetItem(item, makeFacetsContainer, maxMakeCount, 'make'));
        facetResults.transmission.forEach(item => renderFacetItem(item, transmissionFacetsContainer, maxTransmissionCount, 'transmission'));
        facetResults.fuelType.forEach(item => renderFacetItem(item, fuelTypeFacetsContainer, maxFuelTypeCount, 'fuelType'));
        
        // Handle model facets visibility and rendering
        if (selectedFacets.make.length === 1 && facetResults.model && facetResults.model.length > 0) {
            modelFacetsSection.style.display = 'block';
            facetResults.model.forEach(item => renderFacetItem(item, modelFacetsContainer, maxModelCount, 'model'));
        } else {
            modelFacetsSection.style.display = 'none';
            // Clear model filters when make changes
            selectedFacets.model = [];
        }
    }
    
    // Add or remove a facet filter
    function toggleFacetFilter(facetType, facetValue) {
        // Check if this facet value is already selected
        const index = selectedFacets[facetType].indexOf(facetValue);
        
        if (index === -1) {
            // Add the facet filter
            selectedFacets[facetType].push(facetValue);
            
            // If we're selecting a different make, clear any model filters
            if (facetType === 'make' && selectedFacets.make.length > 1) {
                selectedFacets.model = [];
            }
        } else {
            // Remove the facet filter
            selectedFacets[facetType].splice(index, 1);
            
            // If we're removing a make and it was the only one, clear model filters
            if (facetType === 'make' && selectedFacets.make.length === 0) {
                selectedFacets.model = [];
            }
        }
        
        // Run a new search with the updated filters
        updateSearchResults();
    }
    
    // Reset conversation
    async function resetConversation() {
        try {
            await fetch('/api/reset', {
                method: 'POST'
            });
            
            // Clear chat
            chatContainer.innerHTML = '';
            
            // Reset to initial state with default search
            manualPreferences = { price: 0 };
            
            // Reset facet filters
            selectedFacets = {
                make: [],
                model: [],
                transmission: [],
                fuelType: []
            };
            
            // Show loading indicator
            loadingOverlay.classList.remove('d-none');
            
            // Display the default preference
            displayPreferences(manualPreferences);
            preferencesPillContainer.classList.remove('d-none');
            
            // Run the search again
            fetchInitialSearch();
            
        } catch (error) {
            console.error('Error resetting conversation:', error);
        }
    }
    
    // Initialize the page
    initialize();
}); 