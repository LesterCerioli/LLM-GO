
from __future__ import annotations

from string import Template

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

_AUTH_SERVICE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> implementations

// services/implementations/auth_service.go
// Pattern: JWT HS256 token generation + validation service
// Observed in: Medical-App-Core/services/implementations/auth_service.go
package implementations

import (
\t"fmt"
\t"os"
\t"time"

\t"github.com/golang-jwt/jwt/v4"
\t"golang.org/x/crypto/bcrypt"
)

// Claims holds the JWT payload.
type Claims struct {
\tUserID string `json:"user_id"`
\tEmail  string `json:"email"`
\tRole   string `json:"role"`
\tjwt.RegisteredClaims
}

// AuthTokenServiceImpl handles JWT creation and validation.
type AuthTokenServiceImpl struct {
\tsecret []byte
\tttl    time.Duration
}

// NewAuthTokenService reads JWT_SECRET from the environment and constructs the service.
func NewAuthTokenService() *AuthTokenServiceImpl {
\tsecret := os.Getenv("JWT_SECRET")
\tif secret == "" {
\t\tpanic("JWT_SECRET environment variable must be set")
\t}
\treturn &AuthTokenServiceImpl{
\t\tsecret: []byte(secret),
\t\tttl:    24 * time.Hour,
\t}
}

// GenerateToken creates a signed HS256 JWT for the given user attributes.
func (s *AuthTokenServiceImpl) GenerateToken(userID, email, role string) (string, error) {
\tclaims := &Claims{
\t\tUserID: userID,
\t\tEmail:  email,
\t\tRole:   role,
\t\tRegisteredClaims: jwt.RegisteredClaims{
\t\t\tExpiresAt: jwt.NewNumericDate(time.Now().Add(s.ttl)),
\t\t\tIssuedAt:  jwt.NewNumericDate(time.Now()),
\t\t\tIssuer:    "medical-sas-api",
\t\t},
\t}
\ttoken := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
\tsigned, err := token.SignedString(s.secret)
\tif err != nil {
\t\treturn "", fmt.Errorf("sign token: %w", err)
\t}
\treturn signed, nil
}

// ValidateToken parses and validates a JWT string. Returns nil on success.
func (s *AuthTokenServiceImpl) ValidateToken(tokenString string) error {
\ttoken, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(t *jwt.Token) (any, error) {
\t\tif _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
\t\t\treturn nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
\t\t}
\t\treturn s.secret, nil
\t})
\tif err != nil {
\t\treturn fmt.Errorf("invalid token: %w", err)
\t}
\tif !token.Valid {
\t\treturn fmt.Errorf("token is not valid")
\t}
\treturn nil
}

// ParseClaims extracts the Claims from a valid JWT string.
func (s *AuthTokenServiceImpl) ParseClaims(tokenString string) (*Claims, error) {
\tclaims := &Claims{}
\t_, err := jwt.ParseWithClaims(tokenString, claims, func(t *jwt.Token) (any, error) {
\t\treturn s.secret, nil
\t})
\tif err != nil {
\t\treturn nil, fmt.Errorf("parse claims: %w", err)
\t}
\treturn claims, nil
}

// HashPassword returns a bcrypt hash of the plain-text password.
func HashPassword(plain string) (string, error) {
\thashed, err := bcrypt.GenerateFromPassword([]byte(plain), bcrypt.DefaultCost)
\tif err != nil {
\t\treturn "", fmt.Errorf("hash password: %w", err)
\t}
\treturn string(hashed), nil
}

// CheckPassword returns nil if plain matches the bcrypt hash.
func CheckPassword(plain, hash string) error {
\tif err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(plain)); err != nil {
\t\treturn fmt.Errorf("invalid credentials")
\t}
\treturn nil
}
</go_file>
""")

_AUTH_MIDDLEWARE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> middleware

// middleware/auth.go
// Pattern: Fiber JWT bearer middleware with claims propagation via Locals
// Observed pattern from: Medical-App-Core/cmd/main.go AuthMiddleware
package middleware

import (
\t"strings"

\t"$module/services/implementations"
\t"github.com/gofiber/fiber/v2"
)

// BearerAuth validates the Authorization: Bearer header and stores parsed
// claims in fiber.Ctx.Locals("claims") for downstream handlers.
func BearerAuth(authSvc *implementations.AuthTokenServiceImpl) fiber.Handler {
\treturn func(c *fiber.Ctx) error {
\t\theader := c.Get("Authorization")
\t\tif !strings.HasPrefix(header, "Bearer ") {
\t\t\treturn c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
\t\t\t\t"error": "missing or malformed Authorization header",
\t\t\t})
\t\t}

\t\ttokenStr := strings.TrimPrefix(header, "Bearer ")
\t\tclaims, err := authSvc.ParseClaims(tokenStr)
\t\tif err != nil {
\t\t\treturn c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
\t\t\t\t"error":  "invalid or expired token",
\t\t\t\t"reason": err.Error(),
\t\t\t})
\t\t}

\t\t// Make claims available to all downstream handlers
\t\tc.Locals("claims", claims)
\t\tc.Locals("user_id", claims.UserID)
\t\tc.Locals("role", claims.Role)
\t\treturn c.Next()
\t}
}

// RoleRequired returns a middleware that allows only the specified roles.
// Must be chained after BearerAuth.
func RoleRequired(roles ...string) fiber.Handler {
\tallowed := make(map[string]struct{}, len(roles))
\tfor _, r := range roles {
\t\tallowed[r] = struct{}{}
\t}
\treturn func(c *fiber.Ctx) error {
\t\trole, _ := c.Locals("role").(string)
\t\tif _, ok := allowed[role]; !ok {
\t\t\treturn c.Status(fiber.StatusForbidden).JSON(fiber.Map{
\t\t\t\t"error": "insufficient permissions",
\t\t\t})
\t\t}
\t\treturn c.Next()
\t}
}
</go_file>
""")

_AUTH_CONTROLLER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> controllers

// controllers/auth_controller.go
// Pattern: login + token generation handler
// Observed in: Medical-App-Core/controllers/auth_controller.go
package controllers

import (
\t"net/http"

\t"$module/services/implementations"
\t"github.com/gofiber/fiber/v2"
)

// AuthController handles token generation and validation endpoints.
type AuthController struct {
\tauthSvc *implementations.AuthTokenServiceImpl
}

func NewAuthController(svc *implementations.AuthTokenServiceImpl) *AuthController {
\treturn &AuthController{authSvc: svc}
}

type generateTokenRequest struct {
\tUserID string `json:"user_id" validate:"required,uuid4"`
\tEmail  string `json:"email"   validate:"required,email"`
\tRole   string `json:"role"    validate:"required"`
}

// GenerateToken godoc
// @Summary     Issue a JWT token
// @Description Generates a signed JWT for the given user credentials
// @Tags        Auth
// @Accept      json
// @Produce     json
// @Param       body body generateTokenRequest true "Token request"
// @Success     200 {object} fiber.Map{token=string}
// @Failure     400 {object} fiber.Map
// @Failure     500 {object} fiber.Map
// @Router      /auth/token [post]
func (ac *AuthController) GenerateToken(c *fiber.Ctx) error {
\tvar req generateTokenRequest
\tif err := c.BodyParser(&req); err != nil {
\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "invalid request payload"})
\t}
\tif req.UserID == "" || req.Email == "" || req.Role == "" {
\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "user_id, email, and role are required"})
\t}

\ttoken, err := ac.authSvc.GenerateToken(req.UserID, req.Email, req.Role)
\tif err != nil {
\t\treturn c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": "failed to generate token"})
\t}

\treturn c.Status(http.StatusOK).JSON(fiber.Map{"token": token})
}

// ValidateToken godoc
// @Summary     Validate a JWT token
// @Description Returns 200 if the token is valid, 401 otherwise
// @Tags        Auth
// @Accept      json
// @Produce     json
// @Param       body body fiber.Map{token=string} true "Token to validate"
// @Success     200 {object} fiber.Map{valid=bool}
// @Failure     401 {object} fiber.Map
// @Router      /auth/validate [post]
func (ac *AuthController) ValidateToken(c *fiber.Ctx) error {
\tvar body struct {
\t\tToken string `json:"token"`
\t}
\tif err := c.BodyParser(&body); err != nil || body.Token == "" {
\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "token is required"})
\t}

\tif err := ac.authSvc.ValidateToken(body.Token); err != nil {
\t\treturn c.Status(http.StatusUnauthorized).JSON(fiber.Map{
\t\t\t"valid":  false,
\t\t\t"reason": err.Error(),
\t\t})
\t}

\treturn c.Status(http.StatusOK).JSON(fiber.Map{"valid": true})
}
</go_file>
""")

_VALIDATOR_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> initializers

// initializers/validators.go
// Pattern: custom validator registration with go-playground/validator
// Observed in: Medical-App-Core/initializers/validators.go
package initializers

import (
\t"strconv"
\t"strings"

\t"github.com/go-playground/validator/v10"
)

// RegisterCustomValidators adds domain-specific validation rules.
func RegisterCustomValidators(v *validator.Validate) {
\t_ = v.RegisterValidation("cpf",  validateCPF)
\t_ = v.RegisterValidation("cnpj", validateCNPJ)
\t_ = v.RegisterValidation("crm",  validateCRM)
}

// validateCPF validates a Brazilian CPF number (11 digits, checksum).
func validateCPF(fl validator.FieldLevel) bool {
\tcpf := strings.ReplaceAll(fl.Field().String(), ".", "")
\tcpf  = strings.ReplaceAll(cpf, "-", "")
\tif len(cpf) != 11 || allEqual(cpf) {
\t\treturn false
\t}
\treturn cpfChecksum(cpf)
}

func cpfChecksum(cpf string) bool {
\tdigits := make([]int, 11)
\tfor i, ch := range cpf {
\t\tdigits[i], _ = strconv.Atoi(string(ch))
\t}
\tfor pass := 0; pass < 2; pass++ {
\t\tweight := 10 + pass
\t\tsum    := 0
\t\tfor i := 0; i < 9+pass; i++ {
\t\t\tsum += digits[i] * (weight - i)
\t\t}
\t\trem := (sum * 10) % 11
\t\tif rem == 10 || rem == 11 {
\t\t\trem = 0
\t\t}
\t\tif rem != digits[9+pass] {
\t\t\treturn false
\t\t}
\t}
\treturn true
}

// validateCNPJ validates a Brazilian CNPJ number (14 digits, checksum).
func validateCNPJ(fl validator.FieldLevel) bool {
\tcnpj := strings.Map(func(r rune) rune {
\t\tif r >= '0' && r <= '9' { return r }
\t\treturn -1
\t}, fl.Field().String())
\tif len(cnpj) != 14 || allEqual(cnpj) {
\t\treturn false
\t}
\treturn cnpjChecksum(cnpj)
}

func cnpjChecksum(cnpj string) bool {
\tweights1 := []int{5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}
\tweights2 := []int{6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}
\tfor pass, weights := range [][]int{weights1, weights2} {
\t\tsum := 0
\t\tfor i, w := range weights {
\t\t\td, _ := strconv.Atoi(string(cnpj[i]))
\t\t\tsum += d * w
\t\t}
\t\trem := sum % 11
\t\texpected := byte('0')
\t\tif rem >= 2 {
\t\t\texpected = byte('0' + (11 - rem))
\t\t}
\t\tif cnpj[12+pass] != expected {
\t\t\treturn false
\t\t}
\t}
\treturn true
}

// validateCRM validates a Brazilian CRM number (digits optionally followed by UF code).
func validateCRM(fl validator.FieldLevel) bool {
\tcrm := fl.Field().String()
\treturn len(crm) >= 4 && len(crm) <= 10
}

func allEqual(s string) bool {
\tif s == "" { return false }
\tfor _, c := range s[1:] {
\t\tif byte(c) != s[0] { return false }
\t}
\treturn true
}
</go_file>
""")


class AuthPatternGenerator:
    """Generate JWT auth and validation training examples."""

    MODULE = "medical-sas-api"

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        for ver in GO_VERSIONS:
            examples.append(_AUTH_SERVICE_TEMPLATE.substitute(go_version=ver))
            examples.append(_AUTH_MIDDLEWARE_TEMPLATE.substitute(go_version=ver, module=self.MODULE))
            examples.append(_AUTH_CONTROLLER_TEMPLATE.substitute(go_version=ver, module=self.MODULE))
            examples.append(_VALIDATOR_TEMPLATE.substitute(go_version=ver))
        return examples
