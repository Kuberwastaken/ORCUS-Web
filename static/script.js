document.getElementById('analyze').addEventListener('click', async () => {
    const comment = document.getElementById('comment').value;
    document.getElementById('loading-bar-container').style.display = 'block'; // Show loading bar
    document.getElementById('result').style.display = 'none'; // Hide result initially

    const response = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment })
    });

    // Simulate loading bar progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        document.getElementById('loading-bar').style.width = `${progress}%`;
        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 100);

    const result = await response.json();
    setTimeout(() => {
        document.getElementById('loading-bar-container').style.display = 'none'; // Hide loading bar
        document.getElementById('result').innerHTML = `
            <h2>Analysis Report</h2>
            <p style="white-space: pre-wrap;">${result.funny_comment}</p>
        `;
        document.getElementById('result').style.display = 'block'; // Show result after loading
    }, 1200); // Wait for loading bar to complete
});
