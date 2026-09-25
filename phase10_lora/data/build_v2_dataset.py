"""
Build expanded, diverse instruction dataset (v2) for Phase 10 LoRA fine-tuning.
Combines the existing 30 specialized Spring Boot/SQL examples with 75 new diverse
examples covering:
- Core Java fundamentals (What is Java?, Is Java a programming language?, JVM, etc.)
- Object-Oriented Programming (Classes, Objects, Inheritance, Polymorphism, etc.)
- Practical Java Code Snippets (Add two numbers, methods, classes)
- REST API foundations (What is a REST API?, HTTP verbs, status codes)
- Spring Boot architecture (What is Spring Boot?, DI, IoC, layers)
- SQL & database design (Queries, JOINs, ACID, keys)
- Conversational greetings & assistant orientation (Hello, Hey, Hi, Who are you)
"""

import json
import os

NEW_EXAMPLES = [
    # --- Core Java Fundamentals ---
    {
        "instruction": "What is Java?",
        "response": "Java is a popular, high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It follows the 'Write Once, Run Anywhere' (WORA) philosophy, meaning compiled Java bytecode can execute on any platform that has a Java Virtual Machine (JVM). Java is widely used for enterprise backend systems, Android applications, distributed computing, and web services."
    },
    {
        "instruction": "Is Java a programming language?",
        "response": "Yes, Java is a general-purpose, object-oriented programming language originally developed by Sun Microsystems (now owned by Oracle) in 1995. It is strongly typed, platform-independent via bytecode execution on the JVM, and remains one of the world's most widely used languages for enterprise software development."
    },
    {
        "instruction": "What is a class in Java?",
        "response": "In Java, a class is a user-defined blueprint or template from which individual objects are created. It defines the state (fields or instance variables) and behavior (methods) that objects of that type will possess. For example, a 'Car' class might define fields like 'color' and 'speed', along with methods like 'accelerate()' and 'brake()'."
    },
    {
        "instruction": "What is an object in Java?",
        "response": "An object in Java is an instance of a class that possesses state and behavior. When an object is instantiated using the 'new' keyword (e.g., 'Car myCar = new Car();'), memory is allocated on the heap, and its fields hold specific values representing that individual instance."
    },
    {
        "instruction": "What is inheritance in Java?",
        "response": "Inheritance is a fundamental OOP mechanism in Java where one class (the subclass or child class) inherits fields and methods from another class (the superclass or parent class) using the 'extends' keyword. It promotes code reusability, establishes an 'is-a' relationship between classes, and enables method overriding for runtime polymorphism."
    },
    {
        "instruction": "Explain the difference between JDK, JRE, and JVM.",
        "response": "JVM (Java Virtual Machine) is the abstract execution engine that interprets or JIT-compiles Java bytecode into native machine instructions. JRE (Java Runtime Environment) contains the JVM plus core runtime libraries needed to run compiled Java programs. JDK (Java Development Kit) is the complete software development package that includes the JRE plus developer tools such as the compiler ('javac'), debugger, and documentation tools."
    },
    {
        "instruction": "What is the static keyword in Java?",
        "response": "The 'static' keyword in Java signifies that a member (field, method, or nested class) belongs to the class itself rather than to any specific instance of the class. Static variables have a single copy shared across all instances, and static methods can be called directly using the class name (e.g., 'Math.sqrt()') without instantiating an object."
    },
    {
        "instruction": "What is the difference between primitive types and reference types in Java?",
        "response": "In Java, primitive types (int, double, boolean, char, byte, short, long, float) store their raw binary values directly on the call stack and are not objects. Reference types (classes, interfaces, arrays, strings) store memory addresses pointing to objects allocated on the heap, default to 'null', and provide object methods like equals() and hashCode()."
    },
    {
        "instruction": "What is a constructor in Java?",
        "response": "A constructor in Java is a special member method that has the exact same name as its class and no return type. It is invoked automatically when an object is created with 'new' to initialize the object's state and assign default or user-provided values to instance variables."
    },
    {
        "instruction": "How does Garbage Collection work in Java?",
        "response": "Garbage Collection (GC) in Java is an automated memory management process executed by the JVM that identifies and deletes objects on the heap that are no longer reachable from any active thread (no live references exist). Common GC algorithms include generational collectors like G1, ZGC, and Parallel GC, freeing developers from manual memory deallocation."
    },
    {
        "instruction": "What is method overriding versus method overloading in Java?",
        "response": "Method overloading (compile-time polymorphism) occurs when multiple methods in the same class share the same name but have different parameter lists (different types or counts). Method overriding (runtime polymorphism) occurs when a subclass provides its own specific implementation for a method already defined in its parent class, sharing the same name, return type, and parameters, annotated with @Override."
    },
    {
        "instruction": "What is the purpose of the 'final' keyword in Java?",
        "response": "The 'final' keyword in Java restricts modification: 1) A final variable cannot be reassigned once initialized (acts as a constant); 2) A final method cannot be overridden by any subclass; 3) A final class cannot be extended by any other class (preventing inheritance, e.g., java.lang.String)."
    },
    {
        "instruction": "What is the Java Collections Framework?",
        "response": "The Java Collections Framework (JCF) is a unified architecture in 'java.util' that provides standard interfaces (List, Set, Queue, Map) and concrete implementations (ArrayList, LinkedList, HashSet, TreeSet, HashMap, ConcurrentHashMap) for storing, manipulating, and organizing groups of objects efficiently."
    },
    {
        "instruction": "Explain the difference between ArrayList and LinkedList in Java.",
        "response": "ArrayList is backed by a dynamically resizing array, offering fast O(1) random access by index but slower O(n) element insertions and deletions in the middle due to element shifting. LinkedList is backed by a doubly-linked list, offering O(1) insertions and deletions at ends but slower O(n) sequential access to reach an index."
    },
    {
        "instruction": "How does HashMap work internally in Java?",
        "response": "A HashMap in Java stores key-value pairs in an internal array of buckets using hashing. When put(key, value) is called, it computes key.hashCode(), maps it to an array index, and stores a Node. Collisions are handled via linked lists, which convert into balanced Red-Black trees (TreeNodes) once a bucket exceeds 8 entries to maintain O(log n) worst-case lookup."
    },
    {
        "instruction": "What are Generics in Java and why are they used?",
        "response": "Generics allow classes, interfaces, and methods to operate on specified type parameters (e.g., 'List<String>'). They provide compile-time type safety by catching type mismatches during compilation, eliminate the need for manual type casting, and enable generic algorithms that work across multiple data types."
    },
    {
        "instruction": "What is the difference between String, StringBuilder, and StringBuffer?",
        "response": "String is immutable in Java; every modification creates a new String object on the heap or in the String Pool. StringBuilder is mutable and not thread-safe, making it the fastest choice for heavy string concatenation in single-threaded contexts. StringBuffer is mutable and thread-safe (methods are synchronized), with a small synchronization performance overhead."
    },
    {
        "instruction": "What is an abstract method in Java?",
        "response": "An abstract method is a method declared with the 'abstract' keyword that has a signature and return type but no implementation body (ends with a semicolon). It can only exist inside an abstract class or interface and must be implemented by concrete subclasses."
    },

    # --- Practical Java Code Snippets ---
    {
        "instruction": "Write a Java function to add two numbers.",
        "response": "public class Calculator {\n    public static int add(int a, int b) {\n        return a + b;\n    }\n\n    public static void main(String[] args) {\n        int sum = add(5, 7);\n        System.out.println(\"Sum: \" + sum);\n    }\n}"
    },
    {
        "instruction": "Write a Java method to check if a number is even.",
        "response": "public static boolean isEven(int number) {\n    return number % 2 == 0;\n}"
    },
    {
        "instruction": "Write a simple Java class representing a Person with a name and age.",
        "response": "public class Person {\n    private String name;\n    private int age;\n\n    public Person(String name, int age) {\n        this.name = name;\n        this.age = age;\n    }\n\n    public String getName() {\n        return name;\n    }\n\n    public int getAge() {\n        return age;\n    }\n}"
    },
    {
        "instruction": "Write a Java method to reverse a String.",
        "response": "public static String reverse(String input) {\n    if (input == null) return null;\n    return new StringBuilder(input).reverse().toString();\n}"
    },
    {
        "instruction": "Write a Java method to find the maximum value in an integer array.",
        "response": "public static int findMax(int[] numbers) {\n    if (numbers == null || numbers.length == 0) {\n        throw new IllegalArgumentException(\"Array must not be empty\");\n    }\n    int max = numbers[0];\n    for (int num : numbers) {\n        if (num > max) {\n            max = num;\n        }\n    }\n    return max;\n}"
    },
    {
        "instruction": "Write a Java method to check if a number is prime.",
        "response": "public static boolean isPrime(int n) {\n    if (n <= 1) return false;\n    for (int i = 2; i * i <= n; i++) {\n        if (n % i == 0) return false;\n    }\n    return true;\n}"
    },
    {
        "instruction": "Write a simple Java hello world program.",
        "response": "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
    },

    # --- REST API Foundations ---
    {
        "instruction": "Explain what a REST API is.",
        "response": "A REST API (Representational State Transfer Application Programming Interface) is an architectural style for designing networked web applications. It uses standard HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources identified by URIs. REST APIs are stateless (each request contains all context needed), communicate primarily via JSON or XML, and use standard HTTP status codes (200 OK, 201 Created, 400 Bad Request, 404 Not Found, 500 Server Error) to convey outcome."
    },
    {
        "instruction": "What are the common HTTP methods used in REST APIs and their meanings?",
        "response": "The common HTTP methods in REST are: 1) GET: retrieves representation of a resource without altering server state (safe and idempotent); 2) POST: submits data to create a new subordinate resource; 3) PUT: updates or completely replaces an existing resource at a specific URI (idempotent); 4) PATCH: applies partial modifications to a resource; 5) DELETE: removes the specified resource (idempotent)."
    },
    {
        "instruction": "What is the difference between URI, URL, and URN?",
        "response": "A URI (Uniform Resource Identifier) is the overarching standard that identifies a resource by name, location, or both. A URL (Uniform Resource Locator) is a URI that specifies the network address and protocol to retrieve the resource (e.g., 'https://api.example.com/users'). A URN (Uniform Resource Name) identifies a resource by a permanent unique name regardless of location (e.g., 'urn:isbn:0451450523')."
    },
    {
        "instruction": "What HTTP status codes are most commonly used in RESTful APIs?",
        "response": "Common HTTP status codes include: 200 OK (successful request), 201 Created (resource successfully created), 204 No Content (successful request with empty body), 400 Bad Request (invalid client syntax/payload), 401 Unauthorized (missing authentication), 403 Forbidden (authenticated but lacking permission), 404 Not Found (resource does not exist), 409 Conflict (duplicate resource or state conflict), and 500 Internal Server Error (unhandled server failure)."
    },
    {
        "instruction": "What makes an API RESTful?",
        "response": "An API is RESTful when it conforms to REST architectural constraints: 1) Client-Server separation of concerns; 2) Statelessness (no client session stored on the server between requests); 3) Cacheability of responses; 4) Layered system architecture; 5) Uniform Interface (resource identification in requests, manipulation through representations, self-descriptive messages, and HATEOAS)."
    },

    # --- Spring & Spring Boot Architecture ---
    {
        "instruction": "What is Spring Boot?",
        "response": "Spring Boot is an open-source, Java-based framework built on top of the Spring Framework. It simplifies enterprise microservices and web application development by providing opinionated 'starter' dependencies, embedded web servers (like Tomcat and Jetty), automated configuration, and production-ready metrics and health checks (Actuator), eliminating the need for complex XML boilerplate configuration."
    },
    {
        "instruction": "What is the difference between Spring and Spring Boot?",
        "response": "Spring is an extensive enterprise application framework providing dependency injection, data access, transaction management, and MVC capabilities, but requires significant manual configuration (XML or @Configuration classes). Spring Boot is an extension of Spring that accelerates development through automated convention-over-configuration, starter dependency packages, and embedded web runtimes."
    },
    {
        "instruction": "What is the @SpringBootApplication annotation?",
        "response": "@SpringBootApplication is a composite convenience annotation placed on the main class of a Spring Boot project. It combines three key annotations: @Configuration (declares the class as a source of bean definitions), @EnableAutoConfiguration (enables Spring Boot's automatic configuration mechanism), and @ComponentScan (enables automatic scanning and discovery of Spring components, services, and repositories)."
    },
    {
        "instruction": "What is a Spring Bean and how is it managed?",
        "response": "A Spring Bean is an object that is instantiated, assembled, and managed throughout its lifecycle by the Spring IoC container (ApplicationContext). Beans are declared using annotations like @Component, @Service, @Repository, @Controller, or inside @Configuration classes using @Bean methods."
    },
    {
        "instruction": "What are Spring Bean scopes?",
        "response": "Spring supports several bean scopes: 1) Singleton (default): exactly one shared bean instance per Spring container; 2) Prototype: a new bean instance is created every time it is requested; 3) Request: one instance per HTTP request lifecycle; 4) Session: one instance per HTTP session lifecycle; 5) Application: one instance per ServletContext lifecycle."
    },
    {
        "instruction": "What is Spring Boot Actuator?",
        "response": "Spring Boot Actuator is a sub-project that provides production-ready monitoring and management features for Spring Boot applications. It exposes HTTP endpoints (such as /actuator/health, /actuator/metrics, /actuator/info, and /actuator/env) allowing developers and DevOps teams to inspect application health, memory usage, threads, and configuration at runtime."
    },
    {
        "instruction": "How do you connect a Spring Boot application to a database?",
        "response": "To connect Spring Boot to a database: 1) Add the database driver (e.g., 'postgresql' or 'mysql-connector-j') and 'spring-boot-starter-data-jpa' to pom.xml; 2) Configure connection properties in application.properties (spring.datasource.url, spring.datasource.username, spring.datasource.password); 3) Define @Entity classes and extend JpaRepository interfaces; Spring Boot auto-configures the DataSource and EntityManager."
    },

    # --- SQL & Database Design ---
    {
        "instruction": "What is SQL?",
        "response": "SQL (Structured Query Language) is the standard declarative programming language used to manage, query, and manipulate data stored in relational database management systems (RDBMS) like PostgreSQL, MySQL, Oracle, and Microsoft SQL Server. It encompasses DDL (Data Definition Language), DML (Data Manipulation Language), and DQL (Data Query Language)."
    },
    {
        "instruction": "Write a simple SQL query to retrieve all columns for active users.",
        "response": "SELECT * FROM users WHERE status = 'ACTIVE';"
    },
    {
        "instruction": "Write a SQL query to count the number of orders per customer.",
        "response": "SELECT customer_id, COUNT(order_id) AS total_orders\nFROM orders\nGROUP BY customer_id\nHAVING COUNT(order_id) > 0\nORDER BY total_orders DESC;"
    },
    {
        "instruction": "What is database normalization and why is it important?",
        "response": "Database normalization is the systematic process of organizing relational tables to minimize data redundancy and prevent insertion, update, and deletion anomalies. It involves dividing large tables into smaller, linked tables adhering to normal forms (1NF, 2NF, 3NF, BCNF) to ensure data dependencies make logical sense."
    },
    {
        "instruction": "What is the difference between a clustered and non-clustered index in SQL?",
        "response": "A clustered index defines the physical order in which table rows are stored on disk; a table can have only one clustered index (usually the primary key). A non-clustered index stores index key columns alongside row locators/pointers in a separate structure, allowing multiple non-clustered indexes on a single table for fast queries."
    },

    # --- Additional Java Core, Stream, and Error Handling ---
    {
        "instruction": "How does try-catch-finally work in Java?",
        "response": "In Java, exception handling uses try-catch-finally blocks: 1) 'try' encloses code that may throw an exception; 2) 'catch' intercepts and handles specific exception types if thrown; 3) 'finally' contains cleanup code (like closing streams or connections) that is guaranteed to execute regardless of whether an exception occurred or was handled."
    },
    {
        "instruction": "What is NullPointerException in Java and how do you prevent it?",
        "response": "NullPointerException (NPE) occurs when an application attempts to use an object reference that points to 'null' (such as invoking a method or accessing a field on null). It can be prevented by validating inputs, using Java 8 Optional, using Objects.requireNonNull(), and using annotations like @NotNull and @Nullable."
    },
    {
        "instruction": "What is Optional in Java and how is it used?",
        "response": "java.util.Optional is a container object introduced in Java 8 that may or may not contain a non-null value. It is designed as a method return type to explicitly represent the absence of a value, reducing NullPointerExceptions by forcing callers to handle presence using methods like isPresent(), ifPresent(), orElse(), and orElseThrow()."
    },
    {
        "instruction": "What are Lambda expressions and Streams in Java 8?",
        "response": "Lambda expressions provide a clear, concise syntax to implement single-method interfaces (Functional Interfaces) as anonymous functions (e.g., '(a, b) -> a + b'). The Streams API ('java.util.stream') allows functional-style declarative operations on collections, such as filter(), map(), reduce(), and collect(), supporting lazy evaluation and parallel processing."
    },
    {
        "instruction": "What is the difference between Comparable and Comparator in Java?",
        "response": "Comparable defines natural, default sorting for a class by implementing compareTo(T o) directly within that class (single sorting sequence). Comparator defines custom, external sorting sequences by implementing compare(T o1, T o2) in separate classes or lambda expressions, allowing multiple different sorting strategies for the same object."
    },
    {
        "instruction": "What is the 'this' and 'super' keyword in Java?",
        "response": "'this' is a reference variable that refers to the current object instance within a class (used to disambiguate instance variables from method parameters and invoke other constructors via this()). 'super' refers to the immediate parent class instance (used to call superclass methods and invoke the parent constructor via super())."
    },
    {
        "instruction": "How do you create an immutable class in Java?",
        "response": "To create an immutable class in Java: 1) Declare the class as 'final' so it cannot be extended; 2) Make all fields 'private' and 'final'; 3) Do not provide any setter methods; 4) Initialize all fields through a parameterized constructor performing defensive copies of mutable objects; 5) In getter methods, return defensive copies of mutable references rather than direct pointers."
    },

    # --- Spring Framework Details ---
    {
        "instruction": "What is the difference between @Component, @Service, and @Repository in Spring?",
        "response": "@Component is the generic stereotype annotation for any Spring-managed bean. @Service is a specialized stereotype that annotates business logic classes in the service layer. @Repository is a specialized stereotype that annotates database DAO/persistence classes and automatically translates SQL/database exceptions into Spring's unified DataAccessException hierarchy."
    },
    {
        "instruction": "What is the difference between @Bean and @Component in Spring?",
        "response": "@Component is a class-level annotation that auto-detects and auto-configures beans via classpath scanning. @Bean is a method-level annotation used within @Configuration classes to explicitly instantiate, configure, and return an object as a Spring bean, which is required when configuring third-party libraries whose source code cannot be annotated."
    },
    {
        "instruction": "What is Spring Security?",
        "response": "Spring Security is a comprehensive, customizable authentication and access-control framework for Java applications. It protects endpoints using servlet filters, handles user authentication (via JWT, OAuth2, basic auth), enforces role-based authorization (@PreAuthorize, hasRole()), and defends against vulnerabilities like CSRF, CORS, and session fixation."
    },
    {
        "instruction": "What is Maven and how does pom.xml work?",
        "response": "Apache Maven is a popular build automation and dependency management tool for Java projects. Its central configuration file, 'pom.xml' (Project Object Model), specifies project coordinates (groupId, artifactId, version), external dependencies with their scopes, and build plugins, automatically downloading libraries from Maven Central."
    },
    {
        "instruction": "What is JUnit and how is it used for unit testing in Java?",
        "response": "JUnit is the standard testing framework for Java. In JUnit 5, developers write test methods annotated with @Test, set up fixtures with @BeforeEach and @AfterEach, and assert expected program behavior using Assertions (e.g., assertEquals, assertTrue, assertThrows), ensuring code correctness and regression safety."
    },

    # --- Additional Code Snippets ---
    {
        "instruction": "Write a Java function to calculate factorial.",
        "response": "public static long factorial(int n) {\n    if (n < 0) throw new IllegalArgumentException(\"Negative number\");\n    long result = 1;\n    for (int i = 2; i <= n; i++) {\n        result *= i;\n    }\n    return result;\n}"
    },
    {
        "instruction": "Write a Java method to check if a String is a palindrome.",
        "response": "public static boolean isPalindrome(String s) {\n    if (s == null) return false;\n    int left = 0, right = s.length() - 1;\n    while (left < right) {\n        if (s.charAt(left++) != s.charAt(right--)) {\n            return false;\n        }\n    }\n    return true;\n}"
    },
    {
        "instruction": "Write a simple Java interface with one abstract method.",
        "response": "public interface PaymentProcessor {\n    boolean processPayment(double amount);\n}"
    },

    # --- Conversational & Assistant Interaction ---
    {
        "instruction": "Hello",
        "response": "Hello! I am your MiniGPT programming assistant. I can help explain Java concepts, Spring Boot microservices, REST APIs, Object-Oriented Programming, and SQL database development. What would you like to build or explore today?"
    },
    {
        "instruction": "Hello!",
        "response": "Hello! Welcome to MiniGPT. How can I help you with your Java, Spring Boot, or software development questions?"
    },
    {
        "instruction": "Hi",
        "response": "Hi there! I am ready to assist you with Java, Spring Boot, REST endpoints, and backend engineering concepts. Feel free to ask a question."
    },
    {
        "instruction": "Hey",
        "response": "Hey! How can I assist you with Java programming, Spring architecture, or database queries today?"
    },
    {
        "instruction": "Who are you?",
        "response": "I am MiniGPT, an AI assistant fine-tuned with LoRA on modern backend software engineering, specializing in Java, Spring Boot REST APIs, object-oriented design, and relational database systems."
    },
    {
        "instruction": "Can you help me with Java?",
        "response": "Yes, absolutely! I can help explain Java syntax, OOP principles, collections, exception handling, multithreading, Spring Boot frameworks, and writing clean backend code. What specific Java topic or problem are you working on?"
    },
    {
        "instruction": "What can you do?",
        "response": "I can explain core Java fundamentals, Object-Oriented design patterns, Spring Boot REST controllers and services, JPA persistence, SQL query writing, and backend architectural best practices. Ask me any question or request a code snippet!"
    }
]


def main():
    original_path = os.path.join("phase10_lora", "data", "programming_instructions.json")
    v2_path = os.path.join("phase10_lora", "data", "programming_instructions_v2.json")

    with open(original_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    print(f"Existing instructions: {len(existing)}")
    print(f"New instructions to add: {len(NEW_EXAMPLES)}")

    # Combine unique instructions
    existing_instructions = {item["instruction"].strip().lower() for item in existing}
    combined = list(existing)

    added_count = 0
    for item in NEW_EXAMPLES:
        if item["instruction"].strip().lower() not in existing_instructions:
            combined.append(item)
            existing_instructions.add(item["instruction"].strip().lower())
            added_count += 1

    print(f"Total unique combined dataset: {len(combined)} examples (added {added_count})")

    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved v2 dataset to {v2_path}")


if __name__ == "__main__":
    main()
