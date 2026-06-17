document.addEventListener('DOMContentLoaded', () => {
    const symptomsContainer = document.getElementById('symptoms-container');
    const searchInput = document.getElementById('symptom-search');
    const countValue = document.getElementById('count-value');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const predictedService = document.getElementById('predicted-service');
    const resetBtn = document.getElementById('reset-btn');
    
    // Nouveaux éléments pour les capacités
    const capacitiesContainer = document.getElementById('capacities-container');
    const dischargeBtn = document.getElementById('discharge-btn');
    const saturationWarning = document.getElementById('saturation-warning');

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

    // 1.5. Charger les capacités
    async function loadCapacities() {
        try {
            const response = await fetch('/api/capacities');
            if (!response.ok) throw new Error('Erreur réseau');
            const data = await response.json();
            
            if (data.capacities) {
                renderCapacities(data.capacities);
            }
        } catch (error) {
            console.error('Erreur capacités:', error);
            capacitiesContainer.innerHTML = '<div style="color:#ff4444; padding:1rem;">Erreur de chargement des capacités.</div>';
        }
    }

    function renderCapacities(capacities) {
        capacitiesContainer.innerHTML = '';
        capacities.forEach(cap => {
            const percent = (cap.current / cap.max) * 100;
            let statusClass = '';
            if (percent >= 100) statusClass = 'danger';
            else if (percent >= 70) statusClass = 'warning';

            const card = document.createElement('div');
            card.className = `capacity-card ${statusClass}`;
            card.innerHTML = `
                <div class="capacity-name">${cap.service}</div>
                <div class="capacity-bar-bg">
                    <div class="capacity-bar-fill" style="width: ${Math.min(percent, 100)}%"></div>
                </div>
                <div class="capacity-stats">
                    <span>${cap.current} / ${cap.max} lits</span>
                    <span>${Math.min(Math.round(percent), 100)}%</span>
                </div>
            `;
            capacitiesContainer.appendChild(card);
        });
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
        saturationWarning.classList.add('hidden');

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
            
            if (data.is_saturated) {
                saturationWarning.classList.remove('hidden');
            }
            
            resultContainer.classList.remove('hidden');
            
            // Rafraîchir les jauges de capacité
            await loadCapacities();
            
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
        saturationWarning.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // 7. Simuler sorties
    dischargeBtn.addEventListener('click', async () => {
        dischargeBtn.disabled = true;
        dischargeBtn.textContent = 'Libération...';
        try {
            await fetch('/api/discharge', { method: 'POST' });
            await loadCapacities();
        } catch(e) {
            console.error(e);
        } finally {
            dischargeBtn.disabled = false;
            dischargeBtn.textContent = 'Simuler des sorties';
        }
    });

    // Init
    loadSymptoms();
    loadCapacities();
});
