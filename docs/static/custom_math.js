window.MathJax = {
    tex: {
        tags: 'ams',
        inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    options: {
        renderActions: {
            addCopyableMathML: [200, function (doc) { /* Disable MathML copying */ }]
        }
    }
};
