/**
 * Redux store configuration for Atlas AI Mobile App
 */

import {configureStore, combineReducers} from '@reduxjs/toolkit';
import {
  persistStore,
  persistReducer,
  FLUSH,
  REHYDRATE,
  PAUSE,
  PERSIST,
  PURGE,
  REGISTER,
} from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Import reducers
import mapReducer from './slices/mapSlice';
import alertsReducer from './slices/alertsSlice';
import userReducer from './slices/userSlice';
import mediaReducer from './slices/mediaSlice';
import watchlistReducer from './slices/watchlistSlice';
import systemReducer from './slices/systemSlice';

// Root reducer
const rootReducer = combineReducers({
  map: mapReducer,
  alerts: alertsReducer,
  user: userReducer,
  media: mediaReducer,
  watchlist: watchlistReducer,
  system: systemReducer,
});

// Persist configuration
const persistConfig = {
  key: 'atlas-ai-safety',
  version: 1,
  storage: AsyncStorage,
  whitelist: ['user', 'watchlist'], // Only persist user preferences and watchlist
  blacklist: ['map', 'alerts', 'media', 'system'], // Don't persist real-time data
};

// Create persisted reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

// Configure store
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
      immutableCheck: {
        warnAfter: 128,
      },
    }),
  devTools: __DEV__,
});

// Create persistor
export const persistor = persistStore(store);

// Export types
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks for use throughout the app
export type AppStore = typeof store;