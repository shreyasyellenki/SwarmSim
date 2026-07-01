export class SwarmSocket {
  constructor(onConnect, onDisconnect, onMessage) {
    this.onConnect = onConnect;
    this.onDisconnect = onDisconnect;
    this.onMessage = onMessage;
    this.ws = null;
    this.reconnectDelay = 1000;
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${window.location.host}/ws`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.onConnect();
    };

    this.ws.onmessage = (event) => {
      const state = JSON.parse(event.data);
      this.onMessage(state);
    };

    this.ws.onclose = () => {
      this.onDisconnect();
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000);
    };
  }
}
