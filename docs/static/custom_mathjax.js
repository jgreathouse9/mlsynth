window.MathJax = {
    tex: {
        tags: 'ams',
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        macros: {
            // mathtools/unicode commands MathJax doesn't ship by default
            coloneqq: '\\mathrel{:=}'
        }
    },
    options: {
        renderActions: {
            addCopyableMathML: [200, function (doc) { /* Disable MathML copying */ }]
        }
    }
};
