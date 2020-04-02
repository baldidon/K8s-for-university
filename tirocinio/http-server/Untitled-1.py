#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
# Creiamo la classe che riceverà e risponderà alla richieste HTTP
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    # Implementiamo il metodo che risponde alle richieste GET
    def do_GET(self):
    # Specifichiamo il codice di risposta   
        self.send_response(200)
        # Specifichiamo uno o più header
        self.send_header('Content-type','text/html')
        self.end_headers()
        # Specifichiamo il messaggio che costituirà il corpo della risposta
        #hostname = socket.gethostname()
        message = f"Hello world dal pod "#{hostname}"
        self.wfile.write(bytes(message, "utf8"))
        return
    

print('Avvio del server...')
# Specifichiamo le impostazioni del server
# Scegliamo la porta 8081 (per la porta 80 sono necessari i permessi di root)
server_address = ('0.0.0.0', 8080)
httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
print('Server in esecuzione...')
print('sulla porta 8080')
#httpd.serve_forever()
    


