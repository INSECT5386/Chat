// index.js (메인 진입점)
const figlet = require('figlet');
const { startChatInterface } = require('./api-0');

const API_URL = 'https://yuchan5386-api-0.hf.space/chat';

// 실행 시 예쁜 ASCII 아트 로고 출력
figlet('S3GeN Chat', (err, data) => {
  if (err) {
    console.log('로고 생성 실패...');
    console.dir(err);
    return;
  }
  console.log(data);
  // 로고 찍힌 뒤에 채팅 인터페이스 시작
  startChatInterface(API_URL);
});
