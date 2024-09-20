document.addEventListener('DOMContentLoaded', function () {
    const toggleThemeButton = document.getElementById('toggleTheme');

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }

    toggleThemeButton.addEventListener('click', () => {
        const currentTheme = localStorage.getItem('theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(newTheme);
    });

    // Apply the saved theme on page load
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
});
