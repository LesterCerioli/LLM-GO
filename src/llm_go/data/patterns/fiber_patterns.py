
from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Domain vocabulary used to generate diverse, realistic examples
# ---------------------------------------------------------------------------

DOMAINS = [
    dict(
        domain="medical",
        module="medical-sas-api",
        entity="Patient",
        entity_lower="patient",
        entity_plural="patients",
        entity_snake="patient",
        id_field="CPF",
        id_param="cpf",
        service_contract="PatientServiceContract",
        service_field="PatientService",
        dto="PatientDTO",
        input_dto="PatientInputDTO",
        tag="Patients",
        extra_fields='if input.Name == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "name is required"})\n\t}\n\tif input.Contact == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "contact is required"})\n\t}',
    ),
    dict(
        domain="ecommerce",
        module="ecommerce-api",
        entity="Order",
        entity_lower="order",
        entity_plural="orders",
        entity_snake="order",
        id_field="OrderNumber",
        id_param="order_number",
        service_contract="OrderServiceContract",
        service_field="OrderService",
        dto="OrderDTO",
        input_dto="OrderInputDTO",
        tag="Orders",
        extra_fields='if input.CustomerID == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "customer_id is required"})\n\t}\n\tif len(input.Items) == 0 {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "items cannot be empty"})\n\t}',
    ),
    dict(
        domain="hrm",
        module="hrm-api",
        entity="Employee",
        entity_lower="employee",
        entity_plural="employees",
        entity_snake="employee",
        id_field="CPF",
        id_param="cpf",
        service_contract="EmployeeServiceContract",
        service_field="EmployeeService",
        dto="EmployeeDTO",
        input_dto="EmployeeInputDTO",
        tag="Employees",
        extra_fields='if input.DepartmentName == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "department_name is required"})\n\t}\n\tif input.Position == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "position is required"})\n\t}',
    ),
    dict(
        domain="logistics",
        module="logistics-api",
        entity="Shipment",
        entity_lower="shipment",
        entity_plural="shipments",
        entity_snake="shipment",
        id_field="TrackingCode",
        id_param="tracking_code",
        service_contract="ShipmentServiceContract",
        service_field="ShipmentService",
        dto="ShipmentDTO",
        input_dto="ShipmentInputDTO",
        tag="Shipments",
        extra_fields='if input.OriginAddress == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "origin_address is required"})\n\t}\n\tif input.DestinationAddress == "" {\n\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "destination_address is required"})\n\t}',
    ),
]

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_CONTROLLER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> controllers

// controllers/${entity_lower}Controller.go
// Pattern: struct controller + constructor injection + Swagger annotations
// Observed in: Medical-App-Core/controllers/
package controllers

import (
\t"net/http"
\t"time"

\t"$module/services/contracts"
\t"github.com/gofiber/fiber/v2"
)

// ${entity}Controller handles HTTP requests for the ${entity} resource.
type ${entity}Controller struct {
\t${service_field} contracts.${service_contract}
}

// New${entity}Controller constructs a ${entity}Controller with its service dependency.
func New${entity}Controller(svc contracts.${service_contract}) *${entity}Controller {
\treturn &${entity}Controller{${service_field}: svc}
}

// Create${entity} godoc
// @Summary     Create a new ${entity_lower}
// @Description Create a new ${entity_lower} in the system
// @Tags        ${tag}
// @Accept      json
// @Produce     json
// @Security    BearerAuth
// @Param       body body contracts.${input_dto} true "${entity} data"
// @Success     201  {object} fiber.Map
// @Failure     400  {object} fiber.Map
// @Failure     422  {object} fiber.Map
// @Failure     500  {object} fiber.Map
// @Router      /${entity_plural} [post]
func (ctrl *${entity}Controller) Create${entity}(c *fiber.Ctx) error {
\tvar input contracts.${input_dto}
\tif err := c.BodyParser(&input); err != nil {
\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "invalid request payload"})
\t}

\t${extra_fields}

\tid, err := ctrl.${service_field}.CreateFromInput(input)
\tif err != nil {
\t\treturn c.Status(http.StatusUnprocessableEntity).JSON(fiber.Map{"error": err.Error()})
\t}

\treturn c.Status(http.StatusCreated).JSON(fiber.Map{
\t\t"message": "${entity} created successfully",
\t\t"id":      id,
\t})
}

// GetBy${id_field} godoc
// @Summary     Get ${entity_lower} by ${id_field}
// @Description Retrieve a single ${entity_lower} identified by its ${id_field}
// @Tags        ${tag}
// @Produce     json
// @Security    BearerAuth
// @Param       ${id_param} path string true "${entity} ${id_field}"
// @Success     200 {object} contracts.${dto}
// @Failure     400 {object} fiber.Map
// @Failure     404 {object} fiber.Map
// @Router      /${entity_plural}/{${id_param}} [get]
func (ctrl *${entity}Controller) GetBy${id_field}(c *fiber.Ctx) error {
\tidentifier := c.Params("${id_param}")
\tif identifier == "" {
\t\treturn c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "${id_param} is required"})
\t}

\tresult, err := ctrl.${service_field}.FindBy${id_field}(identifier)
\tif err != nil {
\t\treturn c.Status(http.StatusNotFound).JSON(fiber.Map{"error": err.Error()})
\t}

\treturn c.Status(http.StatusOK).JSON(result)
}

// List${entity}s godoc
// @Summary     List all ${entity_lower}s
// @Description Returns a paginated list of ${entity_lower}s
// @Tags        ${tag}
// @Produce     json
// @Security    BearerAuth
// @Param       page  query int false "Page number (default 1)"
// @Param       limit query int false "Page size (default 20)"
// @Success     200  {array}  contracts.${dto}
// @Failure     500  {object} fiber.Map
// @Router      /${entity_plural} [get]
func (ctrl *${entity}Controller) List${entity}s(c *fiber.Ctx) error {
\tpage  := c.QueryInt("page", 1)
\tlimit := c.QueryInt("limit", 20)
\tif page < 1  { page = 1 }
\tif limit < 1 { limit = 20 }

\tresults, err := ctrl.${service_field}.List(page, limit)
\tif err != nil {
\t\treturn c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": "failed to retrieve ${entity_lower}s"})
\t}

\treturn c.Status(http.StatusOK).JSON(fiber.Map{
\t\t"data":  results,
\t\t"page":  page,
\t\t"limit": limit,
\t})
}
</go_file>
""")

_ROUTER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> main

// cmd/main.go — Fiber router setup
// Pattern: group-based routing + auth middleware + Swagger protection
// Observed in: Medical-App-Core/cmd/main.go
package main

import (
\t"log"
\t"os"

\t"$module/controllers"
\t"$module/initializers"
\t_ "$module/docs"

\t"github.com/gofiber/fiber/v2"
\t"github.com/gofiber/fiber/v2/middleware/basicauth"
\t"github.com/gofiber/fiber/v2/middleware/cors"
\t"github.com/gofiber/fiber/v2/middleware/logger"
\t"github.com/gofiber/swagger"
\t"github.com/joho/godotenv"
)

// bearerAuthMiddleware validates the Authorization: Bearer <token> header.
// Returns 401 if missing or invalid; calls Next() on success.
func bearerAuthMiddleware(svc *initializers.Services) fiber.Handler {
\treturn func(c *fiber.Ctx) error {
\t\theader := c.Get("Authorization")
\t\tif len(header) <= 7 || header[:7] != "Bearer " {
\t\t\treturn c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
\t\t\t\t"error": "missing or malformed Authorization header",
\t\t\t})
\t\t}
\t\tif err := svc.AuthTokenService.ValidateToken(header[7:]); err != nil {
\t\t\treturn c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
\t\t\t\t"error":  "invalid or expired token",
\t\t\t\t"reason": err.Error(),
\t\t\t})
\t\t}
\t\treturn c.Next()
\t}
}

func configureMiddleware(app *fiber.App) {
\tapp.Use(cors.New(cors.Config{
\t\tAllowOrigins: "*",
\t\tAllowMethods: "GET,POST,PUT,PATCH,DELETE,OPTIONS",
\t\tAllowHeaders: "Origin, Content-Type, Accept, Authorization",
\t}))
\tapp.Use(logger.New(logger.Config{
\t\tFormat: `[$${time}] $${status} - $${method} $${path}\\n`,
\t}))
}

func registerRoutes(app *fiber.App, svc *initializers.Services) {
\tauth := app.Group("/auth")
\tauth.Post("/token",    controllers.NewAuthController(svc.AuthTokenService).GenerateToken)
\tauth.Post("/validate", controllers.NewAuthController(svc.AuthTokenService).ValidateToken)

\tprotected := app.Group("/", bearerAuthMiddleware(svc))

\tentityCtrl := controllers.New${entity}Controller(svc.${service_field})
\t${entity_lower}s := protected.Group("/${entity_plural}")
\t${entity_lower}s.Post("/",              entityCtrl.Create${entity})
\t${entity_lower}s.Get("/",               entityCtrl.List${entity}s)
\t${entity_lower}s.Get("/:${id_param}",   entityCtrl.GetBy${id_field})
}

func main() {
\tif err := godotenv.Load(); err != nil {
\t\tlog.Println("warning: .env file not found, using environment")
\t}

\tdb  := initializers.InitialDB()
\tinitializers.RunMigrations(db)
\tsvc := initializers.InitServices(db)

\tapp := fiber.New(fiber.Config{AppName: "$module v1"})
\tconfigureMiddleware(app)

\t// Swagger endpoint protected by basic auth
\tswaggerUser := os.Getenv("SWAGGER_USER")
\tswaggerPass := os.Getenv("SWAGGER_PASSWORD")
\tif swaggerUser == "" || swaggerPass == "" {
\t\tlog.Fatal("SWAGGER_USER and SWAGGER_PASSWORD must be set")
\t}
\tswaggerAuth := basicauth.New(basicauth.Config{
\t\tUsers: map[string]string{swaggerUser: swaggerPass},
\t})
\tapp.Get("/swagger/*", swaggerAuth, swagger.HandlerDefault)

\tregisterRoutes(app, svc)

\tport := os.Getenv("PORT")
\tif port == "" {
\t\tport = "8080"
\t}
\tlog.Printf("server listening on :%s", port)
\tif err := app.Listen(":" + port); err != nil {
\t\tlog.Fatalf("server error: %v", err)
\t}
}
</go_file>
""")

_RESPONSE_UTIL_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> utils

// utils/responseUtil.go
// Pattern: response utility interface + JSON helper
// Observed in: Medical-App-Core/utils/responseUtil.go
package utils

import (
\t"encoding/json"

\t"github.com/gofiber/fiber/v2"
)

// ResponseUtil abstracts JSON response construction for testability.
type ResponseUtil interface {
\tRespondWithJSON(c *fiber.Ctx, code int, payload any) error
\tRespondWithError(c *fiber.Ctx, code int, message string) error
}

// JSONResponseUtil implements ResponseUtil using Fiber's context.
type JSONResponseUtil struct{}

func (ru *JSONResponseUtil) RespondWithJSON(c *fiber.Ctx, code int, payload any) error {
\tbody, err := json.Marshal(payload)
\tif err != nil {
\t\treturn c.Status(fiber.StatusInternalServerError).
\t\t\tJSON(fiber.Map{"error": "failed to marshal response"})
\t}
\tc.Set("Content-Type", "application/json")
\treturn c.Status(code).Send(body)
}

func (ru *JSONResponseUtil) RespondWithError(c *fiber.Ctx, code int, message string) error {
\treturn c.Status(code).JSON(fiber.Map{"error": message})
}
</go_file>
""")


class FiberPatternGenerator:
    """Generate Fiber v2 HTTP pattern training examples."""

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        for d in DOMAINS:
            for ver in GO_VERSIONS:
                examples.append(self._controller(d, ver))
                examples.append(self._router(d, ver))
            examples.append(self._response_util(ver=GO_VERSIONS[-1]))
        return examples

    def _controller(self, d: dict, ver: str) -> str:
        return _CONTROLLER_TEMPLATE.substitute(go_version=ver, **d)

    def _router(self, d: dict, ver: str) -> str:
        return _ROUTER_TEMPLATE.substitute(go_version=ver, **d)

    def _response_util(self, ver: str) -> str:
        return _RESPONSE_UTIL_TEMPLATE.substitute(go_version=ver)
