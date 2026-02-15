package tests

import (
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ExternalDependenciesSuite struct {
	suite.Suite
	settingsFile string
}

func (s *ExternalDependenciesSuite) SetupSuite() {
	settingsFromEnv := strings.TrimSpace(os.Getenv("SETTINGS_FILE"))
	settingsFile := settingsFromEnv
	if settingsFile == "" {
		homeDir, err := os.UserHomeDir()
		require.NoError(s.T(), err)
		settingsFile = filepath.Join(homeDir, ".env")
	}

	s.settingsFile = settingsFile

	_, err := os.Stat(settingsFile)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && settingsFromEnv == "" {
			// If defaulting to $HOME/.env and it doesn't exist, continue.
			return
		}
		require.NoError(s.T(), err)
		return
	}

	err = godotenv.Overload(settingsFile)
	require.NoError(s.T(), err)
}

func (s *ExternalDependenciesSuite) SettingsFile() string {
	return s.settingsFile
}
