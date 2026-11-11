export {
  type EDFAnnotation,
  type EDFHeader,
  EDFParseError,
  type EDFRecording,
  type EDFSignal,
  parseEDF,
} from "./edf";
export {
  createSyntheticEEG,
  type SyntheticEEG,
  type SyntheticEEGOptions,
  type SyntheticStreamOptions,
  startSyntheticStream,
} from "./synthetic";
export {
  connectWebSocket,
  type MessageParser,
  parseSampleMessage,
  type ReconnectOptions,
  type WebSocketConnection,
  type WebSocketConstructor,
  type WebSocketHandlers,
  type WebSocketLike,
  type WebSocketOptions,
} from "./websocket";
