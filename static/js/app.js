document.addEventListener('DOMContentLoaded', () => {
    const symptomsContainer = document.getElementById('symptoms-container');
    const searchInput = document.getElementById('symptom-search');
    const countValue = document.getElementById('count-value');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const predictedService = document.getElementById('predicted-service');
    const resetBtn = document.getElementById('reset-btn');

    let allSymptoms = [];
    let selectedSymptoms = new Set();

    // 1. Charger les symptômes depuis l'API
    async function loadSymptoms() {
        try {
            const response = await fetch('/api/symptoms');
            if (!response.ok) throw new Error('Erreur réseau');
            const data = await response.json();
            
            if (data.symptoms) {
                allSymptoms = data.symptoms;
                renderSymptoms(allSymptoms);
            }
        } catch (error) {
            console.error('Erreur lors du chargement des symptômes:', error);
            symptomsContainer.innerHTML = '<div class="loader" style="color: #ff4444;">Erreur de chargement. Vérifiez que le serveur est lancé.</div>';
        }
    }

    // 2. Afficher les symptômes
    function renderSymptoms(symptomsToRender) {
        symptomsContainer.innerHTML = '';
        
        if (symptomsToRender.length === 0) {
            symptomsContainer.innerHTML = '<div style="width: 100%; text-align: center; color: var(--text-secondary); padding: 2rem;">Aucun symptôme trouvé.</div>';
            return;
        }

        symptomsToRender.forEach(symptom => {
            const chip = document.createElement('button');
            chip.className = 'symptom-chip';
            chip.textContent = symptom.replace(/_/g, ' '); // Formater pour l'affichage
            chip.dataset.value = symptom;

            // Garder l'état de sélection
            if (selectedSymptoms.has(symptom)) {
                chip.classList.add('selected');
            }

            chip.addEventListener('click', () => toggleSymptom(symptom, chip));
            symptomsContainer.appendChild(chip);
        });
    }

    // 3. Gérer la sélection
    function toggleSymptom(symptom, element) {
        if (selectedSymptoms.has(symptom)) {
            selectedSymptoms.delete(symptom);
            element.classList.remove('selected');
        } else {
            selectedSymptoms.add(symptom);
            element.classList.add('selected');
        }
        
        updateActionState();
    }

    function updateActionState() {
        const count = selectedSymptoms.size;
        countValue.textContent = count;
        
        if (count > 0) {
            predictBtn.disabled = false;
        } else {
            predictBtn.disabled = true;
        }
    }

    // 4. Recherche/Filtrage
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = allSymptoms.filter(s => 
            s.toLowerCase().replace(/_/g, ' ').includes(query)
        );
        renderSymptoms(filtered);
    });

    // 5. Prédiction
    predictBtn.addEventListener('click', async () => {
        if (selectedSymptoms.size === 0) return;

        // UI Feedback
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyse en cours...';
        resultContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symptoms: Array.from(selectedSymptoms)
                })
            });

            if (!response.ok) throw new Error('Erreur de prédiction');
            const data = await response.json();
            
            // Afficher le résultat
            predictedService.textContent = data.service;
            resultContainer.classList.remove('hidden');
            
            // Scroll au résultat
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            console.error('Erreur de prédiction:', error);
            alert("Une erreur s'est produite lors de la prédiction.");
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Analyser et Recommander';
        }
    });

    // 6. Réinitialisation
    resetBtn.addEventListener('click', () => {
        selectedSymptoms.clear();
        updateActionState();
        renderSymptoms(allSymptoms);
        searchInput.value = '';
        resultContainer.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Init
    loadSymptoms();
});
