<script>
document.addEventListener('DOMContentLoaded', function() {
  const blockTypes = [
    {selector: 'div.assumption', label: 'Assumption', linkClass: 'assumption-link'},
    {selector: 'div.proof', label: 'Proof', linkClass: 'proof-link'},
    {selector: 'div.problem', label: 'Problem', linkClass: 'problem-link'}
  ];

  blockTypes.forEach(({selector, label, linkClass}) => {
    document.querySelectorAll(selector).forEach((block) => {
      const titleText = block.getAttribute('title');
      const linkUrl = block.getAttribute('data-link') || "#"; 
      const targetUrl = block.getAttribute('data-target'); // link to related statement

      // --- Find closest chapter number ---
      let chapter = 0;
      let prev = block.previousElementSibling;
      while (prev) {
        const chapterSpan = prev.querySelector?.('.chapter-number');
        if (chapterSpan) { chapter = chapterSpan.textContent.trim(); break; }
        prev = prev.previousElementSibling;
      }
      if (chapter == 0) {
        let parent = block.parentElement;
        while (parent) {
          const chapterSpan = parent.querySelector?.('h1 .chapter-number');
          if (chapterSpan) { chapter = chapterSpan.textContent.trim(); break; }
          parent = parent.parentElement;
        }
      }

      // --- Count all blocks of this type in the same chapter ---
      const blocksInChapter = Array.from(document.querySelectorAll(selector)).filter(b => {
        let chap = 0;
        let p = b.previousElementSibling;
        while (p) {
          const cs = p.querySelector?.('.chapter-number');
          if (cs) { chap = cs.textContent.trim(); break; }
          p = p.previousElementSibling;
        }
        if (chap == 0) {
          let parent = b.parentElement;
          while (parent) {
            const cs = parent.querySelector?.('h1 .chapter-number');
            if (cs) { chap = cs.textContent.trim(); break; }
            parent = parent.parentElement;
          }
        }
        return chap == chapter;
      });

      const blockNum = blocksInChapter.indexOf(block) + 1;

      // --- Assign a unique ID to every block ---
      if (!block.id) {
        if (targetUrl) {
          block.id = targetUrl.replace(/^#/, ''); // use data-target if present
        } else {
          // fallback: generate ID from type, chapter, and block number
          block.id = `${label.toLowerCase()}-${chapter}-${blockNum}`;
        }
      }

      // --- Create the main link element ---
      const link = document.createElement('a');
      link.className = linkClass;
      link.href = linkUrl;
      link.textContent = titleText
        ? `${label} ${chapter}.${blockNum} (${titleText})`
        : `${label} ${chapter}.${blockNum}`;

      // --- Optional target link (e.g., "[RESULT]") ---
      let targetLink;
      if (targetUrl) {
        targetLink = document.createElement('a');
        targetLink.className = 'statement-link';
        targetLink.href = targetUrl;
        targetLink.textContent = '[RESULT]';
        targetLink.style.marginLeft = '0.5em';
        targetLink.style.fontStyle = 'italic';
        targetLink.style.fontWeight = '400';
        targetLink.style.color = '#555';
      }

      // --- Insert links at the top of the block ---
      block.insertBefore(link, block.firstChild);
      if (targetLink) block.insertBefore(targetLink, block.firstChild.nextSibling);
    });
  });
});
</script>
