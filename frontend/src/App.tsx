import React, { useState } from 'react';
import {
  Typography, TextField, Button, Box, Switch, FormControlLabel, Paper
} from '@mui/material';
import dayjs from 'dayjs';
import axios from 'axios';

export default function App() {
  const [form, setForm] = useState({
    fecha: dayjs().format('YYYY-MM-DD'),
    hora: dayjs().format('HH:00'),
    temperatura: 22,
    humedad: 55,
    Papa: true,
    Mama: true,
    Ipe: false,
    Javi: true,
  });
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post('http://localhost:8000/predict', {
        ...form,
        temperatura: parseFloat(form.temperatura as any),
        humedad: parseFloat(form.humedad as any),
      });
      setResult(res.data);
    } catch (err) {
      setResult({ error: 'Error en la predicción' });
    }
    setLoading(false);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: `-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif`,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          p: { xs: 2, sm: 5 },
          borderRadius: 7,
          boxShadow: '0 8px 32px 0 rgba(60,60,60,0.08)',
          background: 'rgba(255,255,255,0.95)',
          maxWidth: 400,
          width: '100%',
          backdropFilter: 'blur(8px)',
          border: '1px solid #ececec',
        }}
      >
        <Typography
          variant="h4"
          align="center"
          gutterBottom
          sx={{
            fontWeight: 700,
            letterSpacing: '-0.5px',
            color: '#222',
            mb: 1,
          }}
        >
          Potencia eléctrica
        </Typography>
        <Typography align="center" sx={{ color: '#888', mb: 4, fontSize: 16, fontWeight: 400 }}>
          Predicción de la potencia eléctrica de la vivienda
        </Typography>
        <Box
          component="form"
          onSubmit={handleSubmit}
          sx={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2.5,
          }}
        >
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              label="Fecha"
              type="date"
              name="fecha"
              value={form.fecha}
              onChange={handleChange}
              fullWidth
              InputLabelProps={{ shrink: true }}
              sx={{
                background: '#fff',
                borderRadius: 3,
                input: { textAlign: 'center', fontWeight: 500, letterSpacing: '0.5px' },
                boxShadow: 'none',
                border: '1px solid #ececec',
                transition: 'border 0.2s',
                '&:focus-within': { border: '1.5px solid #007aff' },
              }}
            />
            <TextField
              label="Hora"
              type="time"
              name="hora"
              value={form.hora}
              onChange={handleChange}
              fullWidth
              InputLabelProps={{ shrink: true }}
              sx={{
                background: '#fff',
                borderRadius: 3,
                input: { textAlign: 'center', fontWeight: 500, letterSpacing: '0.5px' },
                boxShadow: 'none',
                border: '1px solid #ececec',
                transition: 'border 0.2s',
                '&:focus-within': { border: '1.5px solid #007aff' },
              }}
            />
          </Box>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              label="Temperatura (ºC)"
              name="temperatura"
              type="number"
              value={form.temperatura}
              onChange={handleChange}
              fullWidth
              sx={{
                background: '#fff',
                borderRadius: 3,
                input: { textAlign: 'center', fontWeight: 500 },
                boxShadow: 'none',
                border: '1px solid #ececec',
                transition: 'border 0.2s',
                '&:focus-within': { border: '1.5px solid #007aff' },
              }}
            />
            <TextField
              label="Humedad (%)"
              name="humedad"
              type="number"
              value={form.humedad}
              onChange={handleChange}
              fullWidth
              sx={{
                background: '#fff',
                borderRadius: 3,
                input: { textAlign: 'center', fontWeight: 500 },
                boxShadow: 'none',
                border: '1px solid #ececec',
                transition: 'border 0.2s',
                '&:focus-within': { border: '1.5px solid #007aff' },
              }}
            />
          </Box>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              gap: 1,
              background: '#fafbfc',
              borderRadius: 3,
              p: 2,
              border: '1px solid #ececec',
            }}
          >
            <Typography variant="subtitle1" sx={{ color: '#222', fontWeight: 500, mb: 1, fontSize: 15 }}>
              Personas presentes
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, justifyContent: 'space-between' }}>
              <FormControlLabel
                control={<Switch checked={form.Papa} onChange={handleChange} name="Felipe" color="primary" />}
                label="Papa"
                sx={{ m: 0, '.MuiFormControlLabel-label': { fontWeight: 400, color: '#444' } }}
              />
              <FormControlLabel
                control={<Switch checked={form.Mama} onChange={handleChange} name="Esther" color="primary" />}
                label="Mama"
                sx={{ m: 0, '.MuiFormControlLabel-label': { fontWeight: 400, color: '#444' } }}
              />
              <FormControlLabel
                control={<Switch checked={form.Ipe} onChange={handleChange} name="Ipe" color="primary" />}
                label="Ipe"
                sx={{ m: 0, '.MuiFormControlLabel-label': { fontWeight: 400, color: '#444' } }}
              />
              <FormControlLabel
                control={<Switch checked={form.Javi} onChange={handleChange} name="Javi" color="primary" />}
                label="Javi"
                sx={{ m: 0, '.MuiFormControlLabel-label': { fontWeight: 400, color: '#444' } }}
              />
            </Box>
          </Box>
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            disabled={loading}
            sx={{
              mt: 1,
              py: 1.5,
              fontWeight: 600,
              fontSize: 17,
              borderRadius: 4,
              background: loading ? '#bfc9d1' : 'linear-gradient(90deg, #007aff 0%, #4f8cff 100%)',
              boxShadow: 'none',
              textTransform: 'none',
              letterSpacing: 0,
              transition: 'background 0.2s, transform 0.15s',
              '&:hover': {
                background: 'linear-gradient(90deg, #4f8cff 0%, #007aff 100%)',
                transform: 'translateY(-1px) scale(1.01)',
              },
            }}
          >
            {loading ? 'Calculando...' : 'Predecir Consumo'}
          </Button>
        </Box>
        {result && (
          <Box
            sx={{
              mt: 4,
              textAlign: 'center',
              background: '#fff',
              borderRadius: 4,
              p: 3,
              border: '1px solid #ececec',
            }}
          >
            {result.error ? (
              <Typography color="error" sx={{ fontWeight: 500, fontSize: 17 }}>{result.error}</Typography>
            ) : (
              <>
                <Typography variant="h6" sx={{ color: '#007aff', fontWeight: 700, mb: 1, fontSize: 20 }}>Resultado</Typography>
                <Typography sx={{ fontSize: 18, fontWeight: 500, color: '#222', mb: 0.5 }}>Fase 0 [W]: <span style={{ color: '#007aff' }}>{result.fase0.toFixed(2)}</span></Typography>
                <Typography sx={{ fontSize: 18, fontWeight: 500, color: '#222', mb: 0.5 }}>Fase 1 [W]: <span style={{ color: '#007aff' }}>{result.fase1.toFixed(2)}</span></Typography>
                <Typography sx={{ fontSize: 18, fontWeight: 500, color: '#222' }}>Fase 2 [W]: <span style={{ color: '#007aff' }}>{result.fase2.toFixed(2)}</span></Typography>
              </>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
}
