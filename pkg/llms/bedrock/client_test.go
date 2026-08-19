package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type ClientSuite struct {
	suite.Suite
}

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

func (s *ClientSuite) TestLoadAWSConfigUsesStaticCredentials() {
	s.T().Setenv("AWS_ACCESS_KEY_ID", "access-key")
	s.T().Setenv("AWS_SECRET_ACCESS_KEY", "secret-key")
	s.T().Setenv("AWS_SESSION_TOKEN", "session-token")
	s.T().Setenv("AWS_REGION", "us-west-2")
	s.T().Setenv("AWS_PROFILE", "")

	restore := stubAWSConfigLoader(func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		opts := applyLoadOptions(s.T(), optFns)
		s.Equal("us-west-2", opts.Region)
		s.Empty(opts.SharedConfigProfile)
		require.NotNil(s.T(), opts.Credentials)

		creds, err := opts.Credentials.Retrieve(ctx)
		require.NoError(s.T(), err)
		s.Equal("access-key", creds.AccessKeyID)
		s.Equal("secret-key", creds.SecretAccessKey)
		s.Equal("session-token", creds.SessionToken)

		return aws.Config{Region: opts.Region, Credentials: opts.Credentials}, nil
	})
	defer restore()

	cfg, err := loadAWSConfig(context.Background())
	s.Require().NoError(err)
	s.Equal("us-west-2", cfg.Region)
}

func (s *ClientSuite) TestLoadAWSConfigReturnsErrorForIncompleteStaticCredentials() {
	s.T().Setenv("AWS_ACCESS_KEY_ID", "access-key")
	s.T().Setenv("AWS_SECRET_ACCESS_KEY", "")
	s.T().Setenv("AWS_PROFILE", "")

	restore := stubAWSConfigLoader(func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		s.Fail("awsConfigLoader should not be called for incomplete static credentials")
		return aws.Config{}, nil
	})
	defer restore()

	_, err := loadAWSConfig(context.Background())
	s.Require().Error(err)
	s.Contains(err.Error(), "both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required when using key-based auth")
}

func (s *ClientSuite) TestLoadAWSConfigUsesNamedProfile() {
	s.T().Setenv("AWS_ACCESS_KEY_ID", "")
	s.T().Setenv("AWS_SECRET_ACCESS_KEY", "")
	s.T().Setenv("AWS_PROFILE", "dev-profile")
	s.T().Setenv("AWS_REGION", "")

	restore := stubAWSConfigLoader(func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		opts := applyLoadOptions(s.T(), optFns)
		s.Equal(defaultRegion, opts.Region)
		s.Equal("dev-profile", opts.SharedConfigProfile)
		s.Nil(opts.Credentials)
		return aws.Config{Region: opts.Region}, nil
	})
	defer restore()

	cfg, err := loadAWSConfig(context.Background())
	s.Require().NoError(err)
	s.Equal(defaultRegion, cfg.Region)
}

func (s *ClientSuite) TestLoadAWSConfigUsesDefaultCredentialChainFallback() {
	s.T().Setenv("AWS_ACCESS_KEY_ID", "")
	s.T().Setenv("AWS_SECRET_ACCESS_KEY", "")
	s.T().Setenv("AWS_SESSION_TOKEN", "")
	s.T().Setenv("AWS_PROFILE", "")
	s.T().Setenv("AWS_REGION", "eu-central-1")

	restore := stubAWSConfigLoader(func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		opts := applyLoadOptions(s.T(), optFns)
		s.Equal("eu-central-1", opts.Region)
		s.Empty(opts.SharedConfigProfile)
		s.Nil(opts.Credentials)
		return aws.Config{Region: opts.Region}, nil
	})
	defer restore()

	cfg, err := loadAWSConfig(context.Background())
	s.Require().NoError(err)
	s.Equal("eu-central-1", cfg.Region)
}

func stubAWSConfigLoader(
	fn func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error),
) func() {
	previous := awsConfigLoader
	awsConfigLoader = fn
	return func() {
		awsConfigLoader = previous
	}
}

func applyLoadOptions(t *testing.T, optFns []func(*config.LoadOptions) error) config.LoadOptions {
	t.Helper()

	var opts config.LoadOptions
	for _, optFn := range optFns {
		require.NoError(t, optFn(&opts))
	}
	return opts
}
