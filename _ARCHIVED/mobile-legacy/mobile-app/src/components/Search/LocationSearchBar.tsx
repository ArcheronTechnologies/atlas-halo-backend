/**
 * Location search bar component for searching cities, landmarks, and addresses
 */

import React, {useState, useEffect, useRef} from 'react';
import {
  View,
  TextInput,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Text,
  Keyboard,
  Animated,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch, useSelector} from 'react-redux';

import {
  setSearchQuery,
  clearSearchResults,
  searchLocations,
  selectSearchResults,
  selectIsSearching,
} from '@store/slices/mapSlice';
import {LocationSearchResult} from '@types/api';
import {AppDispatch} from '@store/index';

interface LocationSearchBarProps {
  placeholder?: string;
  onLocationSelect: (location: LocationSearchResult) => void;
  style?: any;
  autoFocus?: boolean;
}

const LocationSearchBar: React.FC<LocationSearchBarProps> = ({
  placeholder = "Search for places to check safety...",
  onLocationSelect,
  style,
  autoFocus = false,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const searchResults = useSelector(selectSearchResults);
  const isSearching = useSelector(selectIsSearching);

  // Local state
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Refs
  const inputRef = useRef<TextInput>(null);
  const searchTimeout = useRef<NodeJS.Timeout>();
  const resultsAnimation = useRef(new Animated.Value(0)).current;

  // Handle search input changes
  const handleSearchChange = (text: string) => {
    setQuery(text);
    dispatch(setSearchQuery(text));

    // Clear previous timeout
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    // Debounce search
    if (text.trim().length > 2) {
      searchTimeout.current = setTimeout(() => {
        dispatch(searchLocations(text.trim()));
        setShowResults(true);
        animateResults(true);
      }, 300);
    } else {
      dispatch(clearSearchResults());
      setShowResults(false);
      animateResults(false);
    }
  };

  // Handle location selection
  const handleLocationSelect = (location: LocationSearchResult) => {
    setQuery(location.displayName);
    setShowResults(false);
    animateResults(false);
    Keyboard.dismiss();
    onLocationSelect(location);
    inputRef.current?.blur();
  };

  // Handle focus
  const handleFocus = () => {
    setIsFocused(true);
    if (searchResults.length > 0 && query.length > 2) {
      setShowResults(true);
      animateResults(true);
    }
  };

  // Handle blur
  const handleBlur = () => {
    setIsFocused(false);
    // Delay hiding results to allow for selection
    setTimeout(() => {
      setShowResults(false);
      animateResults(false);
    }, 150);
  };

  // Clear search
  const handleClear = () => {
    setQuery('');
    dispatch(clearSearchResults());
    setShowResults(false);
    animateResults(false);
    inputRef.current?.focus();
  };

  // Animate results dropdown
  const animateResults = (show: boolean) => {
    Animated.timing(resultsAnimation, {
      toValue: show ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (searchTimeout.current) {
        clearTimeout(searchTimeout.current);
      }
    };
  }, []);

  // Show results when search results update
  useEffect(() => {
    if (searchResults.length > 0 && isFocused && query.length > 2) {
      setShowResults(true);
      animateResults(true);
    }
  }, [searchResults, isFocused, query]);

  // Render search result item
  const renderSearchResult = ({item}: {item: LocationSearchResult}) => (
    <TouchableOpacity
      style={styles.resultItem}
      onPress={() => handleLocationSelect(item)}
    >
      <Icon
        name={getLocationIcon(item.type)}
        size={20}
        color="#666"
        style={styles.resultIcon}
      />
      <View style={styles.resultTextContainer}>
        <Text style={styles.resultName}>{item.name}</Text>
        <Text style={styles.resultAddress}>{item.location.address}</Text>
      </View>
      <Icon
        name="arrow-forward"
        size={16}
        color="#999"
      />
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, style]}>
      {/* Search input */}
      <View style={[
        styles.searchInput,
        isFocused && styles.searchInputFocused,
      ]}>
        <Icon
          name="search"
          size={20}
          color="#666"
          style={styles.searchIcon}
        />
        <TextInput
          ref={inputRef}
          style={styles.textInput}
          placeholder={placeholder}
          placeholderTextColor="#999"
          value={query}
          onChangeText={handleSearchChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          autoFocus={autoFocus}
          returnKeyType="search"
          autoCapitalize="words"
          autoCorrect={false}
        />
        {query.length > 0 && (
          <TouchableOpacity
            style={styles.clearButton}
            onPress={handleClear}
          >
            <Icon
              name="close"
              size={20}
              color="#666"
            />
          </TouchableOpacity>
        )}
        {isSearching && (
          <Icon
            name="hourglass-empty"
            size={20}
            color="#666"
            style={styles.loadingIcon}
          />
        )}
      </View>

      {/* Search results dropdown */}
      {showResults && (
        <Animated.View
          style={[
            styles.resultsContainer,
            {
              opacity: resultsAnimation,
              transform: [{
                translateY: resultsAnimation.interpolate({
                  inputRange: [0, 1],
                  outputRange: [-10, 0],
                }),
              }],
            },
          ]}
        >
          <FlatList
            data={searchResults}
            renderItem={renderSearchResult}
            keyExtractor={(item) => item.id}
            style={styles.resultsList}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyResults}>
                <Text style={styles.emptyResultsText}>
                  {isSearching ? 'Searching...' : 'No results found'}
                </Text>
              </View>
            }
          />
        </Animated.View>
      )}
    </View>
  );
};

// Helper function to get appropriate icon for location type
function getLocationIcon(type: string): string {
  switch (type) {
    case 'city':
      return 'location-city';
    case 'landmark':
      return 'place';
    case 'address':
      return 'home';
    case 'poi':
      return 'room';
    default:
      return 'place';
  }
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    zIndex: 1000,
  },
  searchInput: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    borderRadius: 25,
    paddingHorizontal: 16,
    paddingVertical: 12,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
  },
  searchInputFocused: {
    borderColor: '#2196F3',
    borderWidth: 2,
  },
  searchIcon: {
    marginRight: 12,
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    color: '#333',
    paddingVertical: 0,
  },
  clearButton: {
    padding: 4,
    marginLeft: 8,
  },
  loadingIcon: {
    marginLeft: 8,
  },
  resultsContainer: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: 'white',
    borderRadius: 12,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.3,
    shadowRadius: 8,
    maxHeight: 300,
    marginTop: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
  },
  resultsList: {
    maxHeight: 300,
  },
  resultItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.05)',
  },
  resultIcon: {
    marginRight: 12,
  },
  resultTextContainer: {
    flex: 1,
  },
  resultName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  resultAddress: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  emptyResults: {
    padding: 20,
    alignItems: 'center',
  },
  emptyResultsText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});

export default LocationSearchBar;