// src/store/index.js
import { createStore } from 'redux';
import rootReducer from './reducers/rootReducer'; // Import your root reducer

const store = createStore(rootReducer);

export default store;
