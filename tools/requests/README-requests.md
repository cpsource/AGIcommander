The `requests` library in Python provides a rich set of functions for making HTTP requests easily. Below is a categorized overview of its **core functions and features**:

---

## 🌐 **Main Request Functions**

| Function                                             | Description                     |
| ---------------------------------------------------- | ------------------------------- |
| `requests.get(url, **kwargs)`                        | HTTP GET request                |
| `requests.post(url, data=None, json=None, **kwargs)` | HTTP POST request               |
| `requests.put(url, data=None, **kwargs)`             | HTTP PUT request                |
| `requests.delete(url, **kwargs)`                     | HTTP DELETE request             |
| `requests.head(url, **kwargs)`                       | Retrieves headers only          |
| `requests.options(url, **kwargs)`                    | Describes communication options |
| `requests.patch(url, data=None, **kwargs)`           | Partial update of a resource    |

---

## 🛠️ **Common Parameters (`**kwargs`)**

| Parameter                      | Description                          |
| ------------------------------ | ------------------------------------ |
| `params={'key': 'value'}`      | Add query string to URL              |
| `data={'key': 'value'}`        | Form-encoded data for POST/PUT       |
| `json={'key': 'value'}`        | JSON-encoded body                    |
| `headers={'User-Agent': '...}` | Custom HTTP headers                  |
| `cookies={'session': 'abcd'}`  | Set cookies                          |
| `auth=('user', 'pass')`        | Basic HTTP auth                      |
| `timeout=10`                   | Timeout in seconds                   |
| `allow_redirects=True`         | Follow redirects (default for `get`) |
| `stream=True`                  | Lazy-load response body              |

---

## 📦 **Response Object (`requests.Response`)**

| Property / Method | Description                      |
| ----------------- | -------------------------------- |
| `r.status_code`   | HTTP status code (e.g. 200, 404) |
| `r.headers`       | Response headers                 |
| `r.text`          | Decoded content (as string)      |
| `r.content`       | Raw content (bytes)              |
| `r.json()`        | JSON-decoded body                |
| `r.cookies`       | Cookies sent by server           |
| `r.url`           | Final URL (after redirects)      |
| `r.ok`            | True if `status_code` < 400      |

---

## 🔐 **Session Support**

```python
s = requests.Session()
s.get("https://example.com")
```

Useful for persisting cookies, headers, or connection pooling.

---

## 📄 **Example: Simple GET Request**

```python
import requests

url = 'https://api.github.com'
response = requests.get(url)

if response.ok:
    print(response.json())
```

---

## 🧪 Bonus Utilities

| Function                                      | Description                                                        |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `requests.utils.get_encodings_from_content()` | Infer encoding from HTML content                                   |
| `requests.codes`                              | Dictionary of HTTP status codes (e.g., `requests.codes.ok == 200`) |

---


