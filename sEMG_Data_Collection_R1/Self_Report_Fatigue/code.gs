// Code.gs

// Log fatigue to the bound spreadsheet
function logFatigue(level, participantId, clientTs) {
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getActiveSheet();  // use the active sheet
    const serverTs = new Date();
    sheet.appendRow([serverTs, participantId || '', level || '', clientTs || '']);
    return { status: 'ok' };
  } catch (err) {
    // bubble error to client
    throw new Error('Server error logging fatigue: ' + err.message);
  }
}

// Serve the HTML page
function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('Fatigue Check')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}