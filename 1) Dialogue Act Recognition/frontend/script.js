function predictDialogueAct() {
    const text = document.getElementById('inputText').value;
    if (!text) {
        alert('Please enter some text.');
        return;
    }

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.textContent = data.error;
        } else {
            resultDiv.textContent = 'Predicted Dialogue Act(s): ' + data.dialogue_acts.join(', ');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}