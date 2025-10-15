/**
 * Atlas AI Mobile Safety App
 * Main application entry point
 */

import React from 'react';
import {
  SafeAreaProvider,
} from 'react-native-safe-area-context';
import {
  Provider as PaperProvider,
  DefaultTheme,
} from 'react-native-paper';
import {
  Provider as ReduxProvider,
} from 'react-redux';
import {
  PersistGate,
} from 'redux-persist/integration/react';
import {
  NavigationContainer,
} from '@react-navigation/native';

import {store, persistor} from '@store/index';
import AppNavigator from './src/navigation/AppNavigator';
import LoadingScreen from '@screens/LoadingScreen';
import PermissionHandler from '@components/Common/PermissionHandler';

// App theme configuration
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2196F3', // Blue for safety
    secondary: '#FF9800', // Orange for warnings
    error: '#F44336', // Red for danger
    success: '#4CAF50', // Green for safe
    warning: '#FFC107', // Yellow for caution
    surface: '#FFFFFF',
    background: '#F5F5F5',
    text: '#212121',
    placeholder: '#757575',
  },
};

const App: React.FC = () => {
  return (
    <SafeAreaProvider>
      <ReduxProvider store={store}>
        <PersistGate loading={<LoadingScreen />} persistor={persistor}>
          <PaperProvider theme={theme}>
            <NavigationContainer>
              <PermissionHandler>
                <AppNavigator />
              </PermissionHandler>
            </NavigationContainer>
          </PaperProvider>
        </PersistGate>
      </ReduxProvider>
    </SafeAreaProvider>
  );
};

export default App;