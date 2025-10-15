/**
 * Atlas AI Mobile Safety App
 * Entry point for React Native application
 */

import {AppRegistry} from 'react-native';
import App from './App';
import {name as appName} from './package.json';

// Register the main application component
AppRegistry.registerComponent(appName, () => App);