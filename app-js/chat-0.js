// api-0.js (챗봇 클라이언트 기능 모듈화)
const readline = require('readline');
const esModule = require('eventsource');
const EventSource = esModule.EventSource || esModule.default || esModule;

function startChatInterface(apiUrl) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: 'You> ',
  });

  let es = null;
  let accumulatedResponse = '';

  function startChat(message) {
    if (es) es.close();

    const url = `${apiUrl}?message=${encodeURIComponent(message)}`;
    es = new EventSource(url);

    es.onopen = () => console.log('Connected ✅');

    es.onmessage = (event) => {
      if (event.data === '[DONE]') {
        console.log('\n[대화 종료]');
        accumulatedResponse = '';
        es.close();
        es = null;
        rl.prompt();
        return;
      }

      const token = event.data;
      if (!accumulatedResponse) {
        accumulatedResponse = token;
      } else if (/^[,\.!?]$/.test(token)) {
        accumulatedResponse += token;
      } else {
        accumulatedResponse += ' ' + token;
      }

      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      process.stdout.write('Bot: ' + accumulatedResponse);
    };

    es.onerror = (err) => {
      console.error('\nEventSource error:', err);
      if (es) {
        es.close();
        es = null;
      }
      rl.prompt();
    };
  }

  console.log('채팅 시작 (종료하려면 Ctrl+C)');
  rl.prompt();

  rl.on('line', (line) => {
    const message = line.trim();
    if (!message) {
      rl.prompt();
      return;
    }
    console.log(`\nUser: ${message}`);
    startChat(message);
  });
}

module.exports = { startChatInterface };
