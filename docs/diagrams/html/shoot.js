// Render an HTML file to a PNG screenshot of its .card element.
// Usage: node shoot.js input.html output.png
const puppeteer = require('/Users/amitchorasiya/Documents/Quantum/.mmdc/node_modules/puppeteer');
const path = require('path');

(async () => {
  const [inHtml, outPng] = process.argv.slice(2);
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 800, deviceScaleFactor: 2 });
  await page.goto('file://' + path.resolve(inHtml), { waitUntil: 'networkidle0' });
  const el = await page.$('.card');
  await el.screenshot({ path: outPng });
  await browser.close();
  console.log('shot', outPng);
})();
