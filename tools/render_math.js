// Typeset LaTeX with MathJax for the notes build.
//
// stdin:  JSON [{tex, display}, ...]
// stdout: JSON [{svg, width, height, valign}, ...] with sizes in ex units.

const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const doc = mathjax.document('', {
  InputJax: new TeX({ packages: AllPackages }),
  // 'local' keeps each glyph definition inside its own SVG, so the fragments
  // stay self-contained when WeasyPrint rasterises them independently.
  OutputJax: new SVG({ fontCache: 'local' }),
});

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => (input += chunk));
process.stdin.on('end', () => {
  const items = JSON.parse(input);
  const rendered = items.map(({ tex, display }) => {
    const node = doc.convert(tex, { display: !!display, em: 16, ex: 8, containerWidth: 640 });
    const svg = adaptor.firstChild(node);
    if (adaptor.kind(svg) !== 'svg') {
      throw new Error(`MathJax could not typeset: ${tex}`);
    }
    return {
      svg: adaptor.outerHTML(svg),
      width: adaptor.getAttribute(svg, 'width'),
      height: adaptor.getAttribute(svg, 'height'),
      valign: adaptor.getAttribute(svg, 'style') || '',
    };
  });
  process.stdout.write(JSON.stringify(rendered));
});
