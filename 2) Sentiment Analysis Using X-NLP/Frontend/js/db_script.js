document.addEventListener("DOMContentLoaded", function() {
    const dashboardData = JSON.parse(localStorage.getItem('dashboardData'));
    if (dashboardData) {
        document.getElementById('totalCount').innerText = dashboardData.total_count;
        document.getElementById('posCount').innerText = dashboardData.pos_count;
        document.getElementById('neuCount').innerText = dashboardData.neu_count;
        document.getElementById('negCount').innerText = dashboardData.neg_count;
        document.getElementById('barChart').src = dashboardData.bar_plot;
        document.getElementById('pieChart').src = dashboardData.pie_chart;

        const samplePosContainer = document.getElementById('samplePos');
        dashboardData.sample_text.positive.forEach(text => {
            const p = document.createElement('p');
            p.innerText = text;
            samplePosContainer.appendChild(p);
        });

        const sampleNegContainer = document.getElementById('sampleNeg');
        dashboardData.sample_text.negative.forEach(text => {
            const p = document.createElement('p');
            p.innerText = text;
            sampleNegContainer.appendChild(p);
        });

        const sampleNeuContainer = document.getElementById('sampleNeu');
        dashboardData.sample_text.neutral.forEach(text => {
            const p = document.createElement('p');
            p.innerText = text;
            sampleNeuContainer.appendChild(p);
        });
    }
});