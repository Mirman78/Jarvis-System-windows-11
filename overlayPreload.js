const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('overlayAPI', {
  closeMap: () => ipcRenderer.send('close-map'),
  closeVideo: () => ipcRenderer.send('close-video')
});